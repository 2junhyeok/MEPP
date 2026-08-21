cd ~/ljh/MEPP/OpenRLHF
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

deepspeed --num_gpus 8 --module openrlhf.cli.train_dpo \
  --data.dataset ~/ljh/MEPP/data/openr1_dpo \
  --data.chosen_key chosen \
  --data.rejected_key rejected \
  --data.prompt_key prompt \
  --data.apply_chat_template \
  --data.max_len 8192 \
  --model.beta 0.1 \
  --train.batch_size 16 \
  --train.micro_batch_size 2 \
  --train.max_epochs 3 \
  --model.model_name_or_path Qwen/Qwen2.5-Math-1.5B \
  --ckpt.output_dir ./ckpt/openrlhf_dpo/dpo_ckpt \
  --ckpt.save_steps 500 \
  --ckpt.save_hf \
  --logger.logging_steps 1 \
  --logger.wandb.project "openr1-comparison" \
  --logger.wandb.run_name "qwen2.5-math-1.5b-dpo" \
  --eval.steps 50 \
  --ds.zero_stage 2 \
  --ds.param_dtype bf16 \
  --ds.attn_implementation flash_attention_2 \
  --adam.lr 5e-7 \
  --lr_scheduler cosine \
  --lr_warmup_ratio 0.03 \
  --model.gradient_checkpointing_enable