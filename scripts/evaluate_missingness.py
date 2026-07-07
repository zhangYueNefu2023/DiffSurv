#!/usr/bin/env python
import argparse
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffsurv.config import load_config
from diffsurv.data import make_dataloader, move_batch_to_device
from diffsurv.diffusion import DiffusionSchedule, ddim_sample_drop_replace
from diffsurv.metrics import concordance_index
from diffsurv.model import DiffSurvModel
from diffsurv.utils import safe_torch_load


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def simulated_mask(mask: torch.Tensor, missing_rate: float) -> torch.Tensor:
    keep = (torch.rand_like(mask) >= missing_rate).float()
    return mask * keep


@torch.no_grad()
def evaluate_rate(config, fold: int, missing_rate: float, device: torch.device) -> float:
    data_cfg = config["data"]
    eval_cfg = config["evaluation"]
    model_cfg = config["model"]
    loader, dims, cancer_to_id = make_dataloader(
        processed_dir=data_cfg["processed_dir"],
        fold=fold,
        mode="test",
        batch_size=eval_cfg["batch_size"],
        num_workers=eval_cfg.get("num_workers", 0),
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
    checkpoint_path = Path(config["outputs"]["checkpoint_dir"]) / f"diffsurv_fold{fold}.pt"
    checkpoint = safe_torch_load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    schedule = DiffusionSchedule(
        num_train_steps=config["diffusion"]["num_train_steps"],
        beta_start=config["diffusion"]["beta_start"],
        beta_end=config["diffusion"]["beta_end"],
    ).to(device)

    risks, times, events = [], [], []
    for batch in tqdm(loader, desc=f"fold={fold} missing={missing_rate:.2f}", leave=False):
        batch = move_batch_to_device(batch, device)
        mask = simulated_mask(batch["mask"], missing_rate)
        risk, _ = ddim_sample_drop_replace(
            model,
            batch["x"],
            mask,
            batch["label"],
            schedule,
            num_steps=eval_cfg["ddim_steps"],
            eta=eval_cfg.get("ddim_eta", 0.0),
        )
        risks.extend(risk.detach().cpu().view(-1).tolist())
        times.extend(batch["time"].detach().cpu().view(-1).tolist())
        events.extend(batch["event"].detach().cpu().view(-1).tolist())
    return concordance_index(risks, times, events)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate robustness under simulated modality missingness.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--rates", nargs="*", type=float, default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    folds = args.folds if args.folds is not None else list(range(config["data"]["num_folds"]))
    for rate in args.rates:
        scores = [evaluate_rate(config, fold, rate, device) for fold in folds]
        mean_score = sum(scores) / len(scores)
        print(f"Missingness {rate:.2f}: mean C-index = {mean_score:.4f}; folds = {scores}")


if __name__ == "__main__":
    main()
