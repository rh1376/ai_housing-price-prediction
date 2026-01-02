from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import FEATURE_COLUMNS
from src.utils import resolve_repo_path
from src.dl.model import MLPRegressor


def _to_dense_float32(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def load_preprocessor(preprocessor_path: str | Path):
    path = resolve_repo_path(preprocessor_path)
    return joblib.load(path)


def load_dl_bundle(model_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(resolve_repo_path(model_path), map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "input_dim" not in ckpt:
        raise ValueError("Invalid checkpoint format. Expected keys: 'state_dict' and 'input_dim'.")

    input_dim = int(ckpt["input_dim"])
    model = MLPRegressor(input_dim=input_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, input_dim


def predict_single_dl(
    model: torch.nn.Module,
    preprocessor,
    input_data: Dict[str, Any],
    device: torch.device,
) -> float:
    # Keep only required features, in correct order
    row = {k: input_data.get(k) for k in FEATURE_COLUMNS}

    # zipcode must be categorical string
    if row.get("zipcode") is not None:
        row["zipcode"] = str(row["zipcode"])

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    x = preprocessor.transform(df)
    x = _to_dense_float32(x)

    with torch.no_grad():
        x_t = torch.from_numpy(x).to(device)
        pred = model(x_t).cpu().numpy().reshape(-1)[0]

    return float(pred)
