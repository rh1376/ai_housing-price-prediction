from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS
from src.utils import resolve_repo_path

def load_model(model_path: str | Path) -> Any:
    path = Path(model_path)
    if not path.is_absolute():
        path = resolve_repo_path(str(path))
    return joblib.load(path)


def prepare_features(input_data: Dict[str, Any]) -> pd.DataFrame:
    data = {key: input_data.get(key) for key in FEATURE_COLUMNS}
    data["zipcode"] = str(data["zipcode"])
    return pd.DataFrame([data])


def predict_single(model: Any, input_data: Dict[str, Any]) -> float:
    features = prepare_features(input_data)
    prediction = model.predict(features)
    return float(prediction[0])
