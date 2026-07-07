#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffsurv.config import load_config
from diffsurv.data import make_dataloader, move_batch_to_device
from diffsurv.diffusion import DiffusionSchedule
from diffsurv.losses import DiffSurvLoss
from diffsurv.model import DiffSurvModel


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_fold(config, fold: int, device: torch.device) -> None:
    train_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    output_dir = Path(config["outputs"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    loader, dims, cancer_to_id = make_dataloader(
        processed_dir=data_cfg["processed_dir"],
        fold=fold,
        mode="train",
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("num_workers", 0),
        data_fraction=train_cfg.get("data_fraction", 1.0),
        seed=train_cfg.get("seed", 42),
    )

    model = DiffSurvModel(
        mrna_dim=dims["mrna"],
        mirna_dim=dims["mirna"],
        protein_dim=dims["protein"],
        num_cancers=len(cancer_to_id),
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg.get("dropout", 0.0),
    ).to(device)

    schedule = DiffusionSchedule(
        num_train_steps=config["diffusion"]["num_train_steps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
    ).to(device)
    criterion = DiffSurvLoss(
        schedule=schedule,
        mse_weight=train_cfg["mse_weight"],
        cox_weight=train_cfg["cox_weight"],
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        use_focal=train_cfg.get("use_focal_cox", True),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"fold={fold} epoch={epoch}", leave=False)
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            total_loss, loss_mse, loss_surv = criterion(model, batch)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip", 1.0))
            optimizer.step()
            running += float(total_loss.detach().cpu())
            pbar.set_postfix(
                {
                    "loss": f"{float(total_loss.detach().cpu()):.4f}",
                    "mse": f"{float(loss_mse.detach().cpu()):.4f}",
                    "surv": f"{float(loss_surv.detach().cpu()):.4f}",
                }
            )
        scheduler.step()
        print(f"Fold {fold} epoch {epoch}: mean loss = {running / max(len(loader), 1):.4f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "fold": fold,
        "dimensions": dims,
        "cancer_to_id": cancer_to_id,
        "config": config,
    }
    path = output_dir / f"diffsurv_fold{fold}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiffSurv.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    os.makedirs(config["outputs"]["checkpoint_dir"], exist_ok=True)
    device = get_device()
    folds = args.folds if args.folds is not None else list(range(config["data"]["num_folds"]))
    print(f"Using device: {device}")
    for fold in folds:
        train_fold(config, fold, device)


if __name__ == "__main__":
    main()
