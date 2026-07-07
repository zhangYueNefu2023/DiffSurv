from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


TensorDict = Dict[str, torch.Tensor]


@dataclass
class DiffusionSchedule:
    """Linear DDPM schedule with helper functions for latent diffusion."""

    num_train_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        beta = torch.linspace(self.beta_start, self.beta_end, self.num_train_steps, dtype=torch.float32)
        if self.device is not None:
            beta = beta.to(self.device)
        self.beta = beta
        self.alpha = 1.0 - beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def to(self, device: torch.device) -> "DiffusionSchedule":
        return DiffusionSchedule(self.num_train_steps, self.beta_start, self.beta_end, device)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar.to(x0.device)[t.long()].view(-1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        return x_t, noise


def _extract(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return values.to(t.device)[t.long()].view(-1, *([1] * (len(shape) - 1)))


@torch.no_grad()
def ddim_sample_drop_replace(
    model,
    x_known: TensorDict,
    mask: torch.Tensor,
    cancer_id: torch.Tensor,
    schedule: DiffusionSchedule,
    num_steps: int = 25,
    eta: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DDIM sampling with Drop-and-Replace anchoring for observed modalities.

    mask uses order [mRNA, miRNA, protein], with 1 for observed and 0 for missing.
    Observed latent tokens are restored to their corresponding forward-diffused
    state at every reverse step; missing tokens are updated by the denoising model.
    """

    device = cancer_id.device
    x_known = {k: v.to(device) for k, v in x_known.items()}
    mask = mask.to(device).float()
    z_known = model.tokenizer(x_known)
    z_t = torch.randn_like(z_known)
    mask_expanded = mask.unsqueeze(-1)

    step_ids = torch.linspace(
        schedule.num_train_steps - 1, 0, steps=num_steps, device=device
    ).long()

    for i, t_idx in enumerate(step_ids):
        t = torch.full((z_t.shape[0],), int(t_idx.item()), device=device, dtype=torch.long)

        known_t, _ = schedule.q_sample(z_known, t)
        z_t = mask_expanded * known_t + (1.0 - mask_expanded) * z_t

        pred_noise, _ = model.forward_latent(z_t, cancer_id, t.float())
        alpha_bar_t = _extract(schedule.alpha_bar, t, z_t.shape)
        pred_x0 = (z_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)

        if i == len(step_ids) - 1:
            z_prev = pred_x0
            prev_t = torch.zeros_like(t)
        else:
            prev_t = torch.full(
                (z_t.shape[0],), int(step_ids[i + 1].item()), device=device, dtype=torch.long
            )
            alpha_bar_prev = _extract(schedule.alpha_bar, prev_t, z_t.shape)
            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
            )
            direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * pred_noise
            z_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + direction
            if eta > 0:
                z_prev = z_prev + sigma * torch.randn_like(z_prev)

        known_prev = z_known if i == len(step_ids) - 1 else schedule.q_sample(z_known, prev_t)[0]
        z_t = mask_expanded * known_prev + (1.0 - mask_expanded) * z_prev

    z_t = mask_expanded * z_known + (1.0 - mask_expanded) * z_t
    final_t = torch.zeros(z_t.shape[0], device=device, dtype=torch.float32)
    _, risk_score = model.forward_latent(z_t, cancer_id, final_t)
    return risk_score, z_t
