import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import safe_torch_load


MODALITIES = ("mrna", "mirna", "protein")


class NestedPanCancerDataset(Dataset):
    """Dataset backed by preprocessed patient tensor files and fold metadata."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        cancer_to_id: Dict[str, int],
        fold_indices: Dict[str, torch.Tensor],
        metadata_dir: Path,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.cancer_to_id = cancer_to_id
        self.fold_indices = fold_indices
        self.metadata_dir = metadata_dir

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.metadata.iloc[idx]
        tensor_path = Path(str(row["tensor_path"]))
        if not tensor_path.is_absolute():
            tensor_path = self.metadata_dir / tensor_path
        data = safe_torch_load(str(tensor_path), map_location=torch.device("cpu"))

        x = {mod: data[mod][self.fold_indices[mod]] for mod in MODALITIES}
        mask = torch.tensor(
            [float(row[f"has_{mod}"]) for mod in MODALITIES],
            dtype=torch.float32,
        )
        cancer_id = torch.tensor(self.cancer_to_id[str(row["cancer_type"])], dtype=torch.long)

        return {
            "sample_id": str(row.get("sample_id", idx)),
            "x": x,
            "mask": mask,
            "label": cancer_id,
            "time": torch.tensor(float(row["surv_time"]), dtype=torch.float32),
            "event": torch.tensor(float(row["surv_event"]), dtype=torch.float32),
        }


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    batch["x"] = {k: v.to(device) for k, v in batch["x"].items()}
    for key in ("mask", "label", "time", "event"):
        batch[key] = batch[key].to(device)
    return batch


def load_feature_indices(processed_dir: Path, fold: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    with open(processed_dir / "union_features.json", "r", encoding="utf-8") as f:
        union_features = json.load(f)
    with open(processed_dir / "fold_specific_features.json", "r", encoding="utf-8") as f:
        fold_features = json.load(f)[f"fold_{fold}"]

    fold_indices = {}
    dimensions = {}
    for mod in MODALITIES:
        index_map = {feature: i for i, feature in enumerate(union_features[mod])}
        indices = [index_map[feature] for feature in fold_features[mod]]
        fold_indices[mod] = torch.tensor(indices, dtype=torch.long)
        dimensions[mod] = len(indices)
    return fold_indices, dimensions


def make_dataloader(
    processed_dir: str,
    fold: int,
    mode: str,
    batch_size: int,
    num_workers: int = 0,
    data_fraction: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, Dict[str, int], Dict[str, int]]:
    processed_path = Path(processed_dir)
    metadata = pd.read_csv(processed_path / "metadata_nested_cv.csv")
    if mode == "train":
        metadata = metadata[metadata["fold"] != fold].reset_index(drop=True)
        if data_fraction < 1.0:
            metadata = metadata.sample(frac=1.0, random_state=seed)
            metadata = (
                metadata.groupby("cancer_type", group_keys=False)
                .apply(lambda frame: frame.head(max(1, int(len(frame) * data_fraction))))
                .reset_index(drop=True)
            )
    elif mode in {"val", "test"}:
        metadata = metadata[metadata["fold"] == fold].reset_index(drop=True)
    else:
        raise ValueError("mode must be one of: train, val, test")

    cancer_types = sorted(pd.read_csv(processed_path / "metadata_nested_cv.csv")["cancer_type"].unique())
    cancer_to_id = {name: i for i, name in enumerate(cancer_types)}
    fold_indices, dimensions = load_feature_indices(processed_path, fold)
    dataset = NestedPanCancerDataset(metadata, cancer_to_id, fold_indices, processed_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dimensions, cancer_to_id


def save_cancer_mapping(processed_dir: str, output_path: Optional[str] = None) -> Dict[str, int]:
    processed_path = Path(processed_dir)
    metadata = pd.read_csv(processed_path / "metadata_nested_cv.csv")
    cancer_to_id = {name: i for i, name in enumerate(sorted(metadata["cancer_type"].unique()))}
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cancer_to_id, f, indent=2)
    return cancer_to_id
