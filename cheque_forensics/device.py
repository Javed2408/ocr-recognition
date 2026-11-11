"""Torch device selection (CUDA when available, CPU otherwise) and weight loading."""
import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def get_device():
    """CUDA if a GPU build of torch sees one, else CPU. CHEQUE_DEVICE=cpu|cuda overrides."""
    wanted = os.environ.get("CHEQUE_DEVICE", "").lower()
    if wanted == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if wanted == "cuda":
        print("CHEQUE_DEVICE=cuda requested but CUDA is not available; using CPU")
    return torch.device("cpu")


def save_checkpoint(model, path, **meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


class ModelUnavailable(FileNotFoundError):
    """Weights missing or saved for a different architecture: treat the model as untrained."""


def load_checkpoint(model, path):
    """Load weights into `model` and switch it to eval mode. Returns the saved metadata.

    Raises FileNotFoundError so callers can report which model is untrained.
    """
    if not os.path.isfile(path):
        raise ModelUnavailable(f"trained weights not found: {path}")
    ckpt = torch.load(path, map_location=get_device())
    try:
        model.load_state_dict(ckpt["state_dict"])
    except (RuntimeError, KeyError) as exc:
        raise ModelUnavailable(f"{os.path.basename(path)} does not match the current architecture "
                               f"(retrain it): {str(exc)[:200]}") from exc
    model.to(get_device()).eval()
    return ckpt.get("meta", {})
