from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean, normalize_score, binary_entropy


def aggregate_loss(
    loss: torch.Tensor,
    loss_mask: torch.Tensor,
    token_level_loss: bool = True,
    dp_size: int = 1,
    batch_num_tokens: Optional[float] = None,
    global_batch_size: Optional[float] = None,
) -> torch.Tensor:
    """Aggregate a per-token loss matrix into a scalar using one of two reduction modes:

    - ``token_level_loss=True``  -> per-token: masked-sum / global token count.
    - ``token_level_loss=False`` -> per-sample: sum of per-sequence token-means / global
      sample count.

    ``batch_num_tokens`` (token mode) and ``global_batch_size`` (sample mode) carry the
    *global* batch totals so the loss is invariant to data-parallel sharding; ``dp_size``
    compensates for the gradient averaging that DeepSpeed/DDP applies across DP ranks.
    """
    if token_level_loss:
        if batch_num_tokens is None:
            return masked_mean(loss, loss_mask, dim=None)
        return (loss * loss_mask).sum() / batch_num_tokens * dp_size

    token_counts = loss_mask.sum(dim=-1)
    seq_loss = (loss * loss_mask).sum(dim=-1) / (token_counts + 1e-8)
    seq_mask = (token_counts > 0).float()  # exclude fully masked sequences
    if global_batch_size is None:
        return masked_mean(seq_loss, seq_mask, dim=None)
    return (seq_loss * seq_mask).sum() / global_batch_size * dp_size


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


class SFTLoss(nn.Module):
    """
    SFT Loss
    """

    def __init__(self, token_level_loss: bool = True):
        super().__init__()
        self.token_level_loss = token_level_loss

    def forward(
        self,
        per_token_logps: torch.Tensor,
        loss_mask: torch.Tensor,
        dp_size: int = 1,
        batch_num_tokens: Optional[float] = None,
        global_batch_size: Optional[float] = None,
    ) -> torch.Tensor:
        loss = aggregate_loss(
            -per_token_logps,
            loss_mask,
            token_level_loss=self.token_level_loss,
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
            global_batch_size=global_batch_size,
        )

        return loss


class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        dual_clip: float = None,
        token_level_loss: bool = True,
        policy_loss_type: str = "ppo",
        enable_vllm_is_correction: bool = False,
        vllm_is_truncated_threshold: list = None,
        vllm_is_correction_type: str = "tis",
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip
        self.policy_loss_type = policy_loss_type
        self.enable_vllm_is_correction = enable_vllm_is_correction
        self.vllm_is_truncated_threshold = vllm_is_truncated_threshold
        self.vllm_is_correction_type = vllm_is_correction_type

        # GSPO requires sequence-level loss (per-sample mean)
        if policy_loss_type == "gspo":
            self.token_level_loss = False

        # Dual-clip PPO: https://arxiv.org/pdf/1912.09729
        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"

        if self.vllm_is_correction_type not in {"tis", "icepop", "seq-mask-tis"}:
            raise ValueError(
                f"Invalid vllm_is_correction_type: {self.vllm_is_correction_type}, must be one of tis/icepop/seq-mask-tis"
            )

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        rollout_log_probs: Optional[torch.Tensor] = None,
        dp_size: int = 1,
        batch_num_tokens: Optional[float] = None,
        global_batch_size: Optional[float] = None,
    ) -> torch.Tensor:
        raw_policy_log_ratio = log_probs - old_log_probs
        if self.policy_loss_type == "ppo":
            policy_log_ratio = raw_policy_log_ratio.clamp(min=-20.0, max=20.0)
            ratio = policy_log_ratio.exp()
        elif self.policy_loss_type == "gspo":
            # GSPO: https://arxiv.org/pdf/2507.18071
            if self.enable_vllm_is_correction:
                log_ratio = log_probs - rollout_log_probs
            else:
                log_ratio = raw_policy_log_ratio
            ratio = (log_ratio * action_mask).sum(dim=-1) / action_mask.sum(dim=-1).clamp(min=1)
            ratio = ratio.exp().unsqueeze(-1) * action_mask
        else:
            raise ValueError(f"Invalid policy loss type: {self.policy_loss_type}")

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

        if self.dual_clip is None:
            # Standard PPO
            loss = -torch.min(surr1, surr2)
        else:
            # Standard PPO clipping
            clip1 = torch.min(surr1, surr2)
            # Dual-clip: additional lower bound for negative advantages
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            # Apply dual-clip: use clip2 for negative advantages, clip1 for positive advantages
            loss = -torch.where(advantages < 0, clip2, clip1)

        # Your Efficient RL Framework Secretly Brings You Off-Policy RL Training: https://fengyao.notion.site/off-policy-rl
        vllm_kl = None
        if self.enable_vllm_is_correction and self.policy_loss_type == "ppo":
            low_threshold, high_threshold = self.vllm_is_truncated_threshold
            rollout_log_ratio = old_log_probs - rollout_log_probs
            if self.vllm_is_correction_type == "icepop":
                # ICEPOP: token-level filtering (set coefficients outside the interval to 0)
                vllm_is = torch.exp(rollout_log_ratio).detach()
                mask = (vllm_is >= low_threshold) & (vllm_is <= high_threshold)
                vllm_is = vllm_is * mask
                loss = vllm_is * loss
            elif self.vllm_is_correction_type == "seq-mask-tis":
                # seq-mask-tis: use sequence-level geometric mean only for filtering,
                # correction coefficients still use TIS (token-level clamp)
                seq_log_ratio = masked_mean(rollout_log_ratio, action_mask, dim=-1)
                seq_is = torch.exp(seq_log_ratio)
                seq_mask = (seq_is >= low_threshold) & (seq_is <= high_threshold)
                vllm_is = torch.exp(rollout_log_ratio).detach()
                loss = seq_mask.unsqueeze(-1) * vllm_is * loss
            else:
                # TIS: token-level clamp with low and high thresholds
                vllm_is = torch.exp(rollout_log_ratio).clamp(min=low_threshold, max=high_threshold).detach()
                loss = vllm_is * loss
            vllm_kl = masked_mean(rollout_log_probs - old_log_probs, action_mask, dim=None)

        loss = aggregate_loss(
            loss,
            action_mask,
            token_level_loss=self.token_level_loss,
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
            global_batch_size=global_batch_size,
        )
        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        ppo_kl = masked_mean(-raw_policy_log_ratio.detach(), action_mask, dim=None)
        return loss, clip_ratio, ppo_kl, vllm_kl


class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None, token_level_loss: bool = True) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.token_level_loss = token_level_loss

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        dp_size: int = 1,
        batch_num_tokens: Optional[float] = None,
        global_batch_size: Optional[float] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = aggregate_loss(
            loss,
            action_mask,
            token_level_loss=self.token_level_loss,
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
            global_batch_size=global_batch_size,
        )
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        # pair_entropy
        #print("d_theta:", d_theta_fp32)# [718.7297, -2890.9688]
        #print("q_theta:", q_theta)# [1., 0.]
        #print("pair_entropy:",pair_entropy_value)# [nan, 1.8421e-07]
        
        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        metrics = {
            "logps": {
                "chosen": policy_chosen_logps.detach().float().mean().item(),
                "rejected": policy_rejected_logps.detach().float().mean().item(),
            },
        }
        return loss, chosen_rewards, rejected_rewards, metrics

class MEPPLoss(nn.Module):
    """
    MEPLoss
    """
    def __init__(self, rho: float, score_mode: str = "mean", reduction: str = "mean"):
        super().__init__()
        if not (0.5 <= rho < 1.0):
            raise ValueError(f"rho must be in (0.5, 1), got {rho}.")
        if score_mode not in ("sum", "mean"):
            raise ValueError(f"score_mode must be one of ('sum', 'mean'), got {score_mode!r}.")
        if reduction not in ("sum", "mean", "none"):
            raise ValueError(f"reduction must be one of ('sum', 'mean', 'none'), got {reduction!r}.")
        self.rho = rho
        self.score_mode = score_mode
        self.reduction = reduction
 
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        chosen_lengths: torch.Tensor = None,
        rejected_lengths: torch.Tensor = None,
        reduction: str = None,
    ):
        reduction = reduction or self.reduction
 
        # length-normalize (or not), depending on score_mode
        logp_chosen, logp_rejected = normalize_score(policy_chosen_logps,
                                                                     policy_rejected_logps,
                                                                     chosen_lengths,
                                                                     rejected_lengths,
                                                                     self.score_mode)
        ref_logp_chosen, ref_logp_rejected = normalize_score(reference_chosen_logps,
                                                               reference_rejected_logps,
                                                               chosen_lengths,
                                                               rejected_lengths,
                                                               self.score_mode)
 
        # how much each response's score moved relative to the reference model
        # (this plays the role of DPO's implicit per-response reward, without a beta scale)
        chosen_score_shift = logp_chosen - ref_logp_chosen
        rejected_score_shift = logp_rejected - ref_logp_rejected
        score_shift_gap = chosen_score_shift - rejected_score_shift

        d_theta = logp_chosen - logp_rejected# score gap
        d_theta_fp32 = d_theta.float()
        q_theta = torch.sigmoid(d_theta_fp32)
        
        d_0 = ref_logp_chosen - ref_logp_rejected
        d_0_fp32 = d_0.float()
        q_0 = torch.sigmoid(d_0_fp32)

        rho_tensor = torch.full_like(q_0, self.rho)
        q_star = torch.maximum(q_0, rho_tensor).detach()
 
        per_example_loss = F.binary_cross_entropy_with_logits(d_theta_fp32, q_star, reduction="none")
 
        if reduction == "none":
            loss = per_example_loss
        elif reduction == "mean":
            loss = per_example_loss.mean()
        elif reduction == "sum":
            loss = per_example_loss.sum()
        else:
            raise ValueError(f"Unknown reduction={reduction!r}.")

        residual = (q_star - q_theta).detach()
        pair_entropy_value = binary_entropy(q_theta)
 
        metrics = {
            "chosen_rewards": chosen_score_shift.detach(),
            "reject_rewards": rejected_score_shift.detach(),
            "pair_entropy": pair_entropy_value.detach().float().mean().item(),
            "mepp": {
                "q_theta": q_theta.detach().float().mean().item(),
                "q_0": q_0.detach().float().mean().item(),
                "q_star": q_star.detach().float().mean().item(),
                "residual": residual.float().mean().item(),
                "abs_residual": residual.float().abs().mean().item(),# abs
                "target_at_rho_frac": (q_star <= self.rho + 1e-6).float().mean().item(),
                "already_adq_frac": (q_0 >= self.rho).float().mean().item(),
                "preference_acc": (q_theta > 0.5).float().mean().item(),
                "score_shift_gap": score_shift_gap.detach().float().mean().item(),

                "score_shift_acc": (score_shift_gap.detach() > 0).float().mean().item(),
            },
            "logps": {
                "chosen": policy_chosen_logps.detach().float().mean().item(),
                "rejected": policy_rejected_logps.detach().float().mean().item(),
            },
        }
 
        return loss, metrics