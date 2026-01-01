from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def resolve_repo_path(path_str: str | Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    path = Path(path_str)
    return path if path.is_absolute() else repo_root / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_arff_to_df(data_path: Path) -> pd.DataFrame:
    data, _ = arff.loadarff(data_path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda val: val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else val
            )
    return df


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
