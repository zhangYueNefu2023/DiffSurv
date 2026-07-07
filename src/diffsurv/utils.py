from typing import Any, Optional

import torch


def safe_torch_load(path: str, map_location: Optional[torch.device] = None) -> Any:
    """Load torch files with compatibility across PyTorch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
