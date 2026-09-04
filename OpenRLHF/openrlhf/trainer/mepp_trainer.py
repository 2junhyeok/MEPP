import os
from abc import ABC

from typing import Literal

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models.loss import DPOLoss, MEPPLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
#from openrlhf.utils.utils import pair_entropy
from openrlhf.utils.utils import masked_mean

class MEPPTrainer(ABC):
    """
    Trainer for MEPP(Maximum Entropy Preference Projection) training.
    
    Args:
    """
    def __init__(
        self,
        model,
        ref_model,
        strategy,
        tokenizer,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 0.5,
        max_epochs: int = 3,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.ref_model = ref_model
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.loss_fn = MEPPLoss(
            rho=self.args.model.rho,
            score_mode=self.args.model.score_mode,
            reduction=self.args.model.reduction,
        )
        # Mixtral 8*7b
        self.aux_loss = self.args.model.aux_loss_coef > 1e-8

        # NLL loss
        self.nll_loss = self.args.model.nll_loss_coef > 1e-8

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
        step = consumed_samples // args.train.batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train.batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train.batch_size)

        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )
        acc_sum = 0
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

            self.model.train()
            self.ref_model.eval()
            device = next(self.model.parameters()).device
            
            #train
            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(device)
                c_mask = c_mask.squeeze(1).to(device)
                reject_ids = reject_ids.squeeze(1).to(device)
                r_mask = r_mask.squeeze(1).to(device)
                
                with torch.no_grad():
                    reference_chosen_logps, reference_rejected_logps, _, _ = self.concatenated_forward(
                        self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens, return_entropy=False
                    )
                policy_chosen_logps, policy_rejected_logps, aux_loss, nll_loss, chosen_entropy, rejected_entropy = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens, return_entropy=True
                )

                # for pair entropy
                prompt_lens_tensor = torch.tensor(prompt_id_lens, device=c_mask.device)
                chosen_lengths = (c_mask.float().sum(-1) - prompt_lens_tensor).clamp(min=1)
                rejected_lengths = (r_mask.float().sum(-1) - prompt_lens_tensor).clamp(min=1)
                
                # loss function
                loss, metrics = self.loss_fn(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    chosen_lengths=chosen_lengths,
                    rejected_lengths=rejected_lengths,
                    reduction="mean",
                )
                
                chosen_rewards = metrics["chosen_rewards"]
                reject_rewards = metrics["reject_rewards"]
                pair_entropy = metrics["pair_entropy"]
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0
                # nll loss
                if not self.nll_loss:
                    nll_loss = 0
                
                loss = (
                    loss
                    + aux_loss * self.args.model.aux_loss_coef
                    + nll_loss * self.args.model.nll_loss_coef
                )
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (chosen_rewards > reject_rewards).float().mean().item()
                acc_sum += acc
                loss_sum += loss.item()

                # length normalization
                #policy_chosen_logps_norm = policy_chosen_logps / chosen_lengths
                #policy_rejected_logps_norm = policy_rejected_logps / rejected_lengths
                #pair_entropy_value = pair_entropy(policy_chosen_logps_norm, policy_rejected_logps_norm).item()
                
                logs_dict = {
                    "loss": loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_rewards.float().mean().item(),
                    "reject_reward": reject_rewards.float().mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                    "grad_norm": self.strategy.get_grad_norm(self.model),
                    "pair_entropy": pair_entropy,
                    "chosen_entropy": chosen_entropy,
                    "rejected_entropy": rejected_entropy
                }
                logs_dict.update({f"mepp/{k}": v for k, v in metrics["mepp"].items()})
                logs_dict.update({f"logps/{k}": v for k, v in metrics["logps"].items()})
                
                if self.nll_loss:
                    logs_dict["nll_loss"] = nll_loss.item()
                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    acc_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train.batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
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
        if global_step % args.eval.steps == 0 and self.eval_dataloader is not None:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
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
        self.model.eval()
        self.ref_model.eval()
        with torch.no_grad():
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of global_step %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
            times = 0
            logs_sum = {}
            device = next(self.model.parameters()).device
            for data in eval_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(device)
                c_mask = c_mask.squeeze(1).to(device)
                reject_ids = reject_ids.squeeze(1).to(device)
                r_mask = r_mask.squeeze(1).to(device)
                # reference forward
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,_,
                ) = self.concatenated_forward(
                    self.ref_model,
                    chosen_ids,
                    c_mask,
                    reject_ids,
                    r_mask,
                    prompt_id_lens,
                    return_entropy=False,
                )
                # policy forward
                (
                    policy_chosen_logps,
                    policy_rejected_logps,
                    aux_loss,
                    nll_loss,
                    chosen_entropy,
                    rejected_entropy,
                ) = self.concatenated_forward(
                    self.model,
                    chosen_ids,
                    c_mask,
                    reject_ids,
                    r_mask,
                    prompt_id_lens,
                    return_entropy=True,
                )
                # response length
                prompt_lens_tensor = torch.tensor(prompt_id_lens, device=c_mask.device)
                chosen_lengths = (c_mask.float().sum(-1) - prompt_lens_tensor).clamp(min=1)
                rejected_len = (r_mask.float().sum(-1) - prompt_lens_tensor).clamp(min=1)
                # MEPP loss
                loss, metrics = self.loss_fn(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    chosen_lengths=chosen_lengths,
                    rejected_lengths=rejected_len,
                    reduction="mean",
                )
                chosen_rewards = metrics["chosen_rewards"]
                reject_rewards = metrics["reject_rewards"]
                pair_entropy = metrics["pair_entropy"]

                if not self.aux_loss:
                    aux_loss = 0
                if not self.nll_loss:
                    nll_loss = 0
                loss =(
                    loss
                    + aux_loss * self.args.model.aux_loss_coef
                    + nll_loss * self.args.model.nll_loss_coef
                )
                
                acc = (
                    (chosen_rewards > reject_rewards).float().mean().item()
                )
                logs_dict = {
                    "loss": loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_rewards.float().mean().item(),
                    "reject_reward": reject_rewards.float().mean().item(),
                    "pair_entropy": pair_entropy,
                    "chosen_entropy": chosen_entropy,
                    "rejected_entropy": rejected_entropy,
                }
                logs_dict.update({f"mepp/{k}": v for k, v in metrics["mepp"].items()})
                logs_dict.update({f"logps/{k}": v for k, v in metrics["logps"].items()})
                if self.nll_loss:
                    logs_dict["nll_loss"] = nll_loss.item()
                
                logs_dict = self.strategy.all_reduce(logs_dict)
                for k, v in logs_dict.items():
                    logs_sum[k] = logs_sum.get(k, 0.0) + v
                
                times += 1
                step_bar.set_postfix(logs_dict)
                step_bar.update()
 
            logs_dict = {
                k: v / times for k, v in logs_sum.items()
            }

            logs_dict["loss_mean"] = logs_dict["loss"]
            logs_dict["acc_mean"] = logs_dict["acc"]
            
            step_bar.set_postfix(logs_dict)
            
            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    wandb_logs = {"eval/%s" % k: v for k, v in {**logs_dict, "global_step": steps}.items()}
                    self._wandb.log(wandb_logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens, return_entropy=True):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks, prompt_id_lens = self.concatenated_inputs(
            chosen_ids, c_mask, reject_ids, r_mask, prompt_id_lens
        )

        log_probs, output = model(
            input_ids,
            attention_mask=att_masks,
            return_output=True,
            return_logprobs=True,
            return_entropy = return_entropy, # for token entropy
            ring_attn_group=self.strategy.ring_attn_group,
        )
        
        all_logps_sum, all_logps_mean = self._get_batch_logps(log_probs, att_masks, prompt_id_lens)
        chosen_logps = all_logps_sum[: chosen_ids.shape[0]]
        rejected_logps = all_logps_sum[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        if hasattr(output, "logits"):
            del output.logits
        # token entropy
        if hasattr(output, "entropy") and output.entropy is not None:
            output_entropy = output.entropy# output.entropy [2B, seq_len -1]
            c_resp_mask = att_masks[:chosen_ids.shape[0]].clone()
            r_resp_mask = att_masks[chosen_ids.shape[0]:].clone()
            c_prompt_lens = prompt_id_lens[:chosen_ids.shape[0]]
            r_prompt_lens = prompt_id_lens[chosen_ids.shape[0]:]
            for i, p_len in enumerate(c_prompt_lens):
                c_resp_mask[i, :p_len] = 0

            for i, p_len in enumerate(r_prompt_lens):
                r_resp_mask[i, :p_len] = 0
                
            c_resp_mask = c_resp_mask[:, 1:]
            r_resp_mask = r_resp_mask[:, 1:]
            
            chosen_entropy = masked_mean(output_entropy[:chosen_ids.shape[0]], c_resp_mask).detach().item()
            rejected_entropy = masked_mean(output_entropy[chosen_ids.shape[0]:], r_resp_mask).detach().item()   
        
            return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: chosen_ids.shape[0]].mean(), chosen_entropy, rejected_entropy
        return chosen_logps, rejected_logps, aux_loss, -all_logps_mean[: chosen_ids.shape[0]].mean()
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
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)
        return logprobs_sums, logprobs_means
