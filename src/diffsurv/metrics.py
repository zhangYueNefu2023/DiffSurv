from typing import Iterable

import numpy as np
import torch


def concordance_index(risk_scores: Iterable[float], time: Iterable[float], event: Iterable[float]) -> float:
    """Harrell-style concordance index for risk scores."""

    risk = np.asarray(risk_scores, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    event = np.asarray(event, dtype=bool).reshape(-1)
    concordant = 0.0
    permissible = 0.0
    n = len(time)
    for i in range(n):
        for j in range(i + 1, n):
            if event[i] and time[i] < time[j]:
                permissible += 1.0
                concordant += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
            elif event[j] and time[j] < time[i]:
                permissible += 1.0
                concordant += 1.0 if risk[j] > risk[i] else 0.5 if risk[i] == risk[j] else 0.0
    return float(concordant / permissible) if permissible > 0 else float("nan")


def tensors_to_numpy(*values: torch.Tensor):
    return [value.detach().cpu().numpy() for value in values]
