from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import DiffusionSchedule


def cox_ph_loss(risk_scores: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Cox partial log-likelihood."""

    risk_scores = risk_scores.view(-1)
    time = time.view(-1)
    event = event.float().view(-1)
    order = torch.argsort(time, descending=True)
    risk_scores = risk_scores[order]
    event = event[order]
    log_risk = torch.logcumsumexp(risk_scores, dim=0)
    log_likelihood = risk_scores - log_risk
    return -(log_likelihood * event).sum() / (event.sum() + 1e-8)


def focal_cox_loss(
    risk_scores: torch.Tensor, time: torch.Tensor, event: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    """Focal-weighted Cox partial log-likelihood."""

    risk_scores = risk_scores.view(-1)
    time = time.view(-1)
    event = event.float().view(-1)
    order = torch.argsort(time, descending=True)
    risk_scores = risk_scores[order]
    event = event[order]
    log_risk = torch.logcumsumexp(risk_scores, dim=0)
    log_likelihood = risk_scores - log_risk
    p_event = torch.exp(torch.clamp(log_likelihood, max=0.0))
    focal_weight = (1.0 - p_event).pow(gamma)
    return -((focal_weight * log_likelihood) * event).sum() / (event.sum() + 1e-8)


def masked_mse_loss(
    pred_noise: torch.Tensor, true_noise: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """MSE over observed modality tokens only."""

    mse = F.mse_loss(pred_noise, true_noise, reduction="none").mean(dim=-1)
    mask = mask.float()
    return (mse * mask).sum() / (mask.sum() + 1e-8)


class DiffSurvLoss(nn.Module):
    """Joint masked denoising and survival loss."""

    def __init__(
        self,
        schedule: DiffusionSchedule,
        mse_weight: float = 1.0,
        cox_weight: float = 0.5,
        focal_gamma: float = 2.0,
        use_focal: bool = True,
    ):
        super().__init__()
        self.schedule = schedule
        self.mse_weight = mse_weight
        self.cox_weight = cox_weight
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal

    def forward(self, model, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = batch["label"].device
        z0 = model.tokenizer(batch["x"])

        t = torch.randint(
            low=0,
            high=self.schedule.num_train_steps,
            size=(z0.shape[0],),
            device=device,
            dtype=torch.long,
        )
        z_t, true_noise = self.schedule.q_sample(z0, t)
        pred_noise, pred_risk = model.forward_latent(z_t, batch["label"], t.float())

        loss_mse = masked_mse_loss(pred_noise, true_noise, batch["mask"])
        if self.use_focal:
            loss_surv = focal_cox_loss(pred_risk, batch["time"], batch["event"], self.focal_gamma)
        else:
            loss_surv = cox_ph_loss(pred_risk, batch["time"], batch["event"])

        total = self.mse_weight * loss_mse + self.cox_weight * loss_surv
        return total, loss_mse, loss_surv
