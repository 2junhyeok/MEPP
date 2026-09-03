import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import SFTLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.loss_utils import iter_grad_accum_global_norm
from openrlhf.utils.utils import masked_mean
from openrlhf.utils.utils import pair_entropy

class SFTTrainer(ABC):
    """
    Trainer for supervised fine-tuning (SFT).

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
        batch_size (int, defaults to 1): Batch size for training.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
        disable_ds_ckpt (bool): Whether not to save deepspeed-format model weight. (Deepspeed model weight is used for training recovery)
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt

        self.loss_fn = SFTLoss()

        # Mixtral 8*7b
        self.aux_loss = self.args.model.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.ds.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.logger.wandb.key and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.logger.wandb.key)
            wandb.init(
                entity=strategy.args.logger.wandb.org,
                project=strategy.args.logger.wandb.project,
                group=strategy.args.logger.wandb.group,
                name=strategy.args.logger.wandb.run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.logger.tensorboard_dir and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.logger.tensorboard_dir, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.logger.tensorboard_dir, strategy.args.logger.wandb.run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # Infer num_update_steps_per_epoch from dataloader if not provided
        if num_update_steps_per_epoch is None:
            num_update_steps_per_epoch = len(self.train_dataloader)
        if num_update_steps_per_epoch <= 0:
            raise ValueError(
                f"num_update_steps_per_epoch must be positive, got {num_update_steps_per_epoch}. "
                "Check that your dataset is not smaller than train_batch_size."
            )

        # get eval and save steps
        if args.eval.steps == -1:
            args.eval.steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.ckpt.save_steps == -1:
            args.ckpt.save_steps = float("inf")  # do not save ckpt

        # Restore step and start_epoch
        # step is 1-indexed: the logging check (step % accum_grad == 0) fires at multiples of accum_grad,
        # so +1 ensures we don't re-log the last completed global_step on resume.
        step = consumed_samples // args.train.batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train.batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train.batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            device = next(self.model.parameters()).device

            # Normalize the loss by the global token count of the whole optimizer-step window
            # (not per micro-batch). mask_fn extracts the shifted loss mask -- the same mask
            # aggregate_loss reduces over -- from each (inputs, attention_masks, loss_masks) batch.
            def sft_loss_mask(batch):
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = batch
                c_mask = c_mask.squeeze(1)
                resp_mask = c_mask.clone().bool()
                for i, p_len in enumerate(prompt_id_lens):
                    resp_mask[i, :p_len] = False
                return resp_mask[:, 1:]

            for data, loss_batch_info in iter_grad_accum_global_norm(
                self.train_dataloader, self.strategy, self.strategy.accumulated_gradient, sft_loss_mask
            ):
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(device)
                c_mask = c_mask.squeeze(1).to(device)
                reject_ids = reject_ids.squeeze(1).to(device)
                r_mask = r_mask.squeeze(1).to(device)
                
                input_ids, attention_mask, doubled_prompt_lens = self.concatenated_inputs(
                    chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
                )
                
                per_token_log_probs, output = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_output=True,
                    return_logprobs=True,
                    return_entropy=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )

                # mixtral
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0
                
                
                B = chosen_ids.shape[0]
                
                # chosen loss
                chosen_log_probs = per_token_log_probs[:B]
                chosen_att_mask = attention_mask[:B]
                chosen_loss_mask = torch.zeros_like(chosen_att_mask, dtype=torch.bool)
                chosen_resp_mask = chosen_att_mask.clone().bool()
                for i, p_len in enumerate(prompt_id_lens):
                    chosen_resp_mask[i, :p_len] = False
                chosen_shifted_mask = chosen_resp_mask[:, 1:]
                
                chosen_entropy_value = 0.0
                reject_entropy_value = 0.0
                if hasattr(output, "entropy") and output.entropy is not None:
                    with torch.no_grad():
                        entropy = output.entropy # [2B, T-1]
                        reject_att_mask = attention_mask[B:]
                        reject_resp_mask = reject_att_mask.clone().bool()
                        for i, p_len in enumerate(prompt_id_lens):
                            reject_resp_mask[i, :p_len] = False
                        reject_shifted_mask = reject_resp_mask[:, 1:]
                        
                        chosen_entropy_value = masked_mean(entropy[:B], chosen_shifted_mask).item()
                        reject_entropy_value = masked_mean(entropy[B:], reject_shifted_mask).item()
                
                
                if hasattr(output, "logits"):
                    del output.logits
                
                gpt_loss = self.loss_fn(
                    chosen_log_probs,
                    chosen_shifted_mask,
                    **loss_batch_info,
                )
                loss = gpt_loss + aux_loss * self.args.model.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)




                    
                with torch.no_grad():
                    all_logps_sum, _ = self._get_batch_logps(per_token_log_probs, attention_mask, doubled_prompt_lens)
                    chosen_logps = all_logps_sum[:B]
                    rejected_logps = all_logps_sum[B:]

                    prompt_lens_tensor = torch.tensor(prompt_id_lens, device=c_mask.device)
                    chosen_len = (c_mask.float().sum(-1) - prompt_lens_tensor).clamp(min=1)
                    rejected_len = (r_mask.float().sum(-1) - prompt_lens_tensor).clamp(min=1)

                    chosen_logps_norm = chosen_logps / chosen_len
                    rejected_logps_norm = rejected_logps / rejected_len

                    pair_entropy_value = pair_entropy(chosen_logps_norm, rejected_logps_norm).item()
                
                loss_sum += gpt_loss.item()
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                    "grad_norm": self.strategy.get_grad_norm(self.model),
                    "pair_entropy": pair_entropy_value,
                    "chosen_entropy": chosen_entropy_value,
                    "rejected_entropy": reject_entropy_value,
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train.batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluation
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict=None, client_states=None):
        logs_dict = logs_dict or {}
        client_states = client_states or {}
        if global_step % args.logger.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval.steps == 0:
            # do eval when eval_dataloader is not None and len(dataloader) > 0, avoid zero division in eval.
            if self.eval_dataloader is not None and len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.ckpt.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model, args.ckpt.path, tag, args.ckpt.max_num, args.ckpt.max_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt.path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            chosen_logps_sum = 0.0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            device = next(self.model.parameters()).device
            for data in eval_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(device)
                c_mask = c_mask.squeeze(1).to(device)

                per_token_log_probs, output = self.model(
                    chosen_ids,
                    attention_mask=c_mask,
                    return_output=True,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )
                if hasattr(output, "logits"):
                    del output.logits
                chosen_resp_mask = c_mask.clone().bool()
                for i, p_len in enumerate(prompt_id_lens):
                    chosen_resp_mask[i, :p_len] = False
                chosen_shifted_mask = chosen_resp_mask[:, 1:]

                loss = self.loss_fn(per_token_log_probs, chosen_shifted_mask)

                batch_chosen_logps, _ = self._get_batch_logps(per_token_log_probs, c_mask, prompt_id_lens)
                chosen_logps_sum += batch_chosen_logps.mean().item()
                    
                times += 1
                loss_sum += loss.item()
                bar_dict = {
                    "eval_loss": loss_sum / times,
                    "logps/chosen": chosen_logps_sum / times,
                    }
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks, prompt_id_lens * 2

    def _get_batch_logps(
        self,
        per_token_logps: torch.FloatTensor,
        attention_mask,
        prompt_id_lens,
    ) -> torch.FloatTensor:
        """Compute the summed and averaged log probabilities of the given labels under the given logits.

        Args:
            per_token_logps: Per token log probabilities. Shape: (batch_size, sequence_length)

        Returns:
            (sum, mean) tensors of shape (batch_size,) over the non-masked tokens.
        """
        loss_masks = attention_mask.clone().bool()
        # mask prompts
        for mask, source_len in zip(loss_masks, prompt_id_lens):
            mask[:source_len] = False
        loss_masks = loss_masks[:, 1:]

        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = logprobs_sums / loss_masks.sum(-1)
        return logprobs_sums, logprobs_means