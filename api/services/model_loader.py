"""
Loads the ATBG model checkpoint(s) for the API.

Supports two modes:
  1. Single model  — set MODEL_CHECKPOINT_PATH in .env
  2. Ensemble      — set ENSEMBLE_CHECKPOINT_A, ENSEMBLE_CHECKPOINT_B,
                     and ENSEMBLE_WEIGHT_A (default 0.60) in .env

If no checkpoint exists yet, falls back to an untrained model so the API
still runs end-to-end for development/testing.
"""

import os

import torch

from atbg_model.model import ATBGModel
from api.config import settings

_model_a = None
_model_b = None
_weight_a = 0.60
_model_version = "untrained"
_ensemble_mode = False


def _load_one(path: str) -> ATBGModel | None:
    model = ATBGModel()
    if path and os.path.exists(path):
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint.get("version", os.path.basename(path))
    return None, "missing"


def get_model():
    """Returns (model_a, model_b_or_None, weight_a, version_string, ensemble_mode)"""
    global _model_a, _model_b, _weight_a, _model_version, _ensemble_mode

    if _model_a is not None:
        return _model_a, _model_b, _weight_a, _model_version, _ensemble_mode

    # Check for ensemble env vars
    path_a = os.environ.get("ENSEMBLE_CHECKPOINT_A") or settings.model_checkpoint_path
    path_b = os.environ.get("ENSEMBLE_CHECKPOINT_B", "./checkpoints/real_atbg_final.pt")
    weight_a_env = os.environ.get("ENSEMBLE_WEIGHT_A", "0.60")

    if path_b and os.path.exists(path_b):
        # Ensemble mode
        _ensemble_mode = True
        _weight_a = float(weight_a_env) if weight_a_env else 0.60
        model_a, ver_a = _load_one(path_a)
        model_b, ver_b = _load_one(path_b)
        if model_a is None:
            model_a = ATBGModel()
            model_a.eval()
            ver_a = "untrained"
        if model_b is None:
            model_b = ATBGModel()
            model_b.eval()
            ver_b = "untrained"
        _model_a, _model_b = model_a, model_b
        _model_version = f"ensemble({ver_a},{ver_b},w={_weight_a:.2f})"
    else:
        # Single model mode
        _ensemble_mode = False
        model_a, ver_a = _load_one(path_a)
        if model_a is None:
            model_a = ATBGModel()
            model_a.eval()
            ver_a = "untrained"
        _model_a = model_a
        _model_b = None
        _model_version = ver_a

    return _model_a, _model_b, _weight_a, _model_version, _ensemble_mode


