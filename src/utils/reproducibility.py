"""Reproducibility utilities: random seed management and environment tracking."""

import os
import platform
import random
import sys
from typing import Any, Dict
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_environment_info() -> Dict[str, Any]:
    """Capture software and hardware runtime environment details."""
    cuda_avail = torch.cuda.is_available()
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "cuda_device_count": torch.cuda.device_count() if cuda_avail else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_avail else "N/A",
    }
