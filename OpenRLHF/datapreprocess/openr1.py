"""
Preprocess OpenR1 dataset to parquet format for DPO training
"""

import argparse
import os

import datasets
from transformers import AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dataset_path", 
        default=None,
        help="Path to the raw dataset on local disk. If None, downloads from Hugging Face Hub."
    )
    parser.add_argument(
        "--local_save_dir", 
        default="~/ljh/MEPP/data/openr1_dpo", 
        help="Local directory to save the preprocessed parquet files."
    )
    parser.add_argument(
        "--config_name",
        default="all",
        choices=["default", "extended", "all"],
    )
    parser.add_argument(
        "--include_solution_fallback",
        action="store_true",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Math-1.5B"
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path
    data_source = "open-r1/OpenR1-Math-220k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, args.config_name, split="train")

    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")
            question = question_raw + " " + instruction_following
            answer_raw = example.pop("answer")
            solution_raw = example.pop("solution", None)
            source_raw = example.pop("source", None)
            generations = example.pop("generations")
            correct_flags = example.pop("correctness_math_verify")
            complete_flags = example.pop("is_reasoning_complete")

            complete_gens = [
                (g, ok) for g, ok, complete in zip(generations, correct_flags, complete_flags) if complete
            ]
            correct_gens = [g for g, ok in complete_gens if ok]
            incorrect_gens = [g for g, ok in complete_gens if not ok]

            chosen_source = None
            if correct_gens:
                chosen_content = correct_gens[0]
                chosen_source = "generation"
            elif args.include_solution_fallback and solution_raw:
                chosen_content = solution_raw
                chosen_source = "solution_fallback"
            else:
                chosen_content = None

            rejected_content = incorrect_gens[0] if incorrect_gens else None
            has_pair = chosen_content is not None and rejected_content is not None

            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "chosen": [{"role": "assistant", "content": chosen_content}],
                "rejected": [{"role": "assistant", "content": rejected_content}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "source": source_raw,
                    "num_correct": len(correct_gens),
                    "num_incorrect": len(incorrect_gens),
                    "num_incomplete": len(generations) - len(complete_gens),
                    "chosen_source": chosen_source,
                },
                "is_valid": has_pair,
            }

        return process_fn

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    MAX_LEN = args.max_len

    def is_short_enough(example):
        if not example["is_valid"]:
            return False
        
        chosen_tokens = tokenizer.apply_chat_template(
            example["prompt"] + example["chosen"], tokenize=True
        )
        rejected_tokens = tokenizer.apply_chat_template(
            example["prompt"] + example["rejected"], tokenize=True
        )
        
        return len(chosen_tokens) <= MAX_LEN and len(rejected_tokens) <= MAX_LEN

    train_dataset = train_dataset.map(
        make_map_fn("train"), with_indices=True, remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        make_map_fn("test"), with_indices=True, remove_columns=test_dataset.column_names
    )

    train_dataset = train_dataset.filter(is_short_enough)
    test_dataset = test_dataset.filter(is_short_enough)

    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))