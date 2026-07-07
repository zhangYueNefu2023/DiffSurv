#!/usr/bin/env python
import argparse
import csv
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


@torch.no_grad()
def evaluate_fold(config, fold: int, device: torch.device):
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

    rows = []
    risks, times, events = [], [], []
    for batch in tqdm(loader, desc=f"evaluate fold={fold}", leave=False):
        batch = move_batch_to_device(batch, device)
        risk, _ = ddim_sample_drop_replace(
            model,
            batch["x"],
            batch["mask"],
            batch["label"],
            schedule,
            num_steps=eval_cfg["ddim_steps"],
            eta=eval_cfg.get("ddim_eta", 0.0),
        )
        risk_values = risk.detach().cpu().view(-1).tolist()
        time_values = batch["time"].detach().cpu().view(-1).tolist()
        event_values = batch["event"].detach().cpu().view(-1).tolist()
        for sample_id, r, t, e in zip(batch["sample_id"], risk_values, time_values, event_values):
            rows.append({"fold": fold, "sample_id": sample_id, "risk_score": r, "time": t, "event": e})
        risks.extend(risk_values)
        times.extend(time_values)
        events.extend(event_values)
    return concordance_index(risks, times, events), rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DiffSurv checkpoints.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--output", default="outputs/evaluation_risk_scores.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()
    folds = args.folds if args.folds is not None else list(range(config["data"]["num_folds"]))
    all_rows = []
    for fold in folds:
        c_index, rows = evaluate_fold(config, fold, device)
        all_rows.extend(rows)
        print(f"Fold {fold}: C-index = {c_index:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "sample_id", "risk_score", "time", "event"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved risk scores: {output}")


if __name__ == "__main__":
    main()
