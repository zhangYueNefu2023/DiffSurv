#!/usr/bin/env python
import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


MODALITY_FILES = {
    "mrna": "TCGA-{cancer}.star_tpm.tsv.gz",
    "mirna": "TCGA-{cancer}.mirna.tsv.gz",
    "protein": "TCGA-{cancer}.protein.tsv.gz",
}


def safe_filename(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))


def read_omics(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", index_col=0).T
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df.loc[:, ~df.columns.duplicated()]


def read_survival(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    sample_col = "sample" if "sample" in df.columns else df.columns[0]
    time_col = "OS.time" if "OS.time" in df.columns else "time" if "time" in df.columns else None
    event_col = "OS" if "OS" in df.columns else "event" if "event" in df.columns else None
    if time_col is None or event_col is None:
        raise ValueError(f"Cannot find survival time/event columns in {path}")
    out = df[[sample_col, time_col, event_col]].copy()
    out.columns = ["sample_id", "surv_time", "surv_event"]
    out["surv_time"] = pd.to_numeric(out["surv_time"], errors="coerce")
    out["surv_event"] = pd.to_numeric(out["surv_event"], errors="coerce")
    out = out.dropna(subset=["sample_id", "surv_time", "surv_event"])
    return out.drop_duplicates(subset=["sample_id"], keep="first")


def select_features_by_variance(path: Path, train_patients: Set[str], variance_ratio: float) -> Set[str]:
    df = read_omics(path)
    if df is None:
        return set()
    patients = list(set(df.index) & train_patients)
    if not patients:
        return set()
    df_train = df.loc[patients]
    variances = df_train.var(axis=0).sort_values(ascending=False)
    total = float(variances.sum())
    if total <= 0:
        selected = set(variances.index.tolist())
    else:
        cumulative = variances.cumsum() / total
        k = int((cumulative < variance_ratio).sum()) + 1
        selected = set(variances.head(k).index.tolist())
    del df, df_train
    gc.collect()
    return selected


def build_master_table(raw_data_dir: Path) -> pd.DataFrame:
    records = []
    for cancer_dir in sorted([p for p in raw_data_dir.iterdir() if p.is_dir()]):
        cancer = cancer_dir.name
        surv = read_survival(cancer_dir / f"TCGA-{cancer}.survival.tsv.gz")
        if surv is None:
            continue
        for sample_id in surv["sample_id"].astype(str):
            records.append({"sample_id": sample_id, "cancer_type": cancer})
    if not records:
        raise RuntimeError("No eligible survival records were found. Check --raw-data-dir.")
    return pd.DataFrame(records).drop_duplicates("sample_id").reset_index(drop=True)


def load_and_align(path: Path, features: Iterable[str]) -> Optional[pd.DataFrame]:
    df = read_omics(path)
    if df is None:
        return None
    return df.reindex(columns=list(features), fill_value=0.0)


def build_nested_dataset(args: argparse.Namespace) -> None:
    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    master = build_master_table(raw_data_dir)
    splitter = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    master["fold"] = -1
    for fold, (_, test_idx) in enumerate(splitter.split(master, master["cancer_type"])):
        master.loc[test_idx, "fold"] = fold

    union_features: Dict[str, Set[str]] = {mod: set() for mod in MODALITY_FILES}
    fold_features: Dict[str, Dict[str, List[str]]] = {}

    for fold in range(args.num_folds):
        train_patients = set(master.loc[master["fold"] != fold, "sample_id"].astype(str))
        selected = {mod: set() for mod in MODALITY_FILES}
        for cancer_dir in tqdm(sorted([p for p in raw_data_dir.iterdir() if p.is_dir()]), desc=f"Fold {fold}"):
            cancer = cancer_dir.name
            for mod, pattern in MODALITY_FILES.items():
                selected[mod].update(
                    select_features_by_variance(
                        cancer_dir / pattern.format(cancer=cancer),
                        train_patients,
                        args.variance_ratio,
                    )
                )
        fold_features[f"fold_{fold}"] = {mod: sorted(values) for mod, values in selected.items()}
        for mod in MODALITY_FILES:
            union_features[mod].update(selected[mod])

    union_sorted = {mod: sorted(values) for mod, values in union_features.items()}
    with open(output_dir / "fold_specific_features.json", "w", encoding="utf-8") as f:
        json.dump(fold_features, f, indent=2)
    with open(output_dir / "union_features.json", "w", encoding="utf-8") as f:
        json.dump(union_sorted, f, indent=2)

    patient_fold = dict(zip(master["sample_id"].astype(str), master["fold"].astype(int)))
    records = []
    for cancer_dir in tqdm(sorted([p for p in raw_data_dir.iterdir() if p.is_dir()]), desc="Writing tensors"):
        cancer = cancer_dir.name
        surv = read_survival(cancer_dir / f"TCGA-{cancer}.survival.tsv.gz")
        if surv is None:
            continue
        surv = surv.set_index("sample_id")
        omics = {
            mod: load_and_align(cancer_dir / pattern.format(cancer=cancer), union_sorted[mod])
            for mod, pattern in MODALITY_FILES.items()
        }
        for patient in surv.index.astype(str):
            if patient not in patient_fold:
                continue
            has = {mod: omics[mod] is not None and patient in omics[mod].index for mod in MODALITY_FILES}
            if not any(has.values()):
                continue
            tensor = {}
            for mod in MODALITY_FILES:
                if has[mod]:
                    tensor[mod] = torch.tensor(omics[mod].loc[patient].values, dtype=torch.float32)
                else:
                    tensor[mod] = torch.zeros(len(union_sorted[mod]), dtype=torch.float32)
            tensor_name = f"{safe_filename(patient)}.pt"
            torch.save(tensor, tensor_dir / tensor_name)
            records.append(
                {
                    "sample_id": patient,
                    "cancer_type": cancer,
                    "fold": patient_fold[patient],
                    "surv_time": float(surv.loc[patient, "surv_time"]),
                    "surv_event": float(surv.loc[patient, "surv_event"]),
                    "has_mrna": int(has["mrna"]),
                    "has_mirna": int(has["mirna"]),
                    "has_protein": int(has["protein"]),
                    "tensor_path": str(Path("tensors") / tensor_name),
                }
            )
        del omics
        gc.collect()

    pd.DataFrame(records).to_csv(output_dir / "metadata_nested_cv.csv", index=False)
    print(f"Saved processed dataset to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build nested fold-aware DiffSurv tensors.")
    parser.add_argument("--raw-data-dir", required=True, help="Directory containing one folder per cancer type.")
    parser.add_argument("--output-dir", default="processed_data", help="Output directory for tensors and metadata.")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--variance-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    build_nested_dataset(parse_args())
