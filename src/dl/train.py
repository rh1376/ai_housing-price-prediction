from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CATEGORICAL_FEATURES, DROP_COLUMNS, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from src.dl.model import MLPRegressor
from src.utils import evaluate_model, load_arff_to_df, resolve_repo_path, save_json, seed_everything


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def to_dense_float32(array: object) -> np.ndarray:
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array, dtype=np.float32)


def split_train_val(
    X: np.ndarray, y: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PyTorch MLP for housing price regression.")
    parser.add_argument("--data", required=True, help="Path to dataset.arff")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(features).float()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)
    return total_loss / max(1, total_samples)


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
    return total_loss / max(1, total_samples)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for features, batch_targets in loader:
            features = features.to(device)
            outputs = model(features).cpu().numpy()
            preds.append(outputs)
            targets.append(batch_targets.numpy())
    pred_array = np.vstack(preds).reshape(-1)
    target_array = np.vstack(targets).reshape(-1)
    return target_array, pred_array


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    set_torch_seed(args.seed)

    data_path = resolve_repo_path(args.data)
    models_dir = resolve_repo_path("models") / "dl"
    reports_dir = resolve_repo_path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_arff_to_df(data_path)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    X_train_dense = to_dense_float32(X_train_proc)
    X_test_dense = to_dense_float32(X_test_proc)
    y_train_array = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test_array = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

    X_train_final, X_val_final, y_train_final, y_val_final = split_train_val(
        X_train_dense, y_train_array, args.seed
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_final), torch.from_numpy(y_train_final)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_final), torch.from_numpy(y_val_final)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test_dense), torch.from_numpy(y_test_array)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(input_dim=X_train_final.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_state = None
    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = predict(model, test_loader, device)
    rmse, mae, r2 = evaluate_model(y_true, y_pred)

    save_json(
        {"rmse": rmse, "mae": mae, "r2": r2},
        reports_dir / "dl_metrics.json",
    )
    torch.save(
    {
        "state_dict": model.state_dict(),
        "input_dim": X_train_final.shape[1],
    },
    models_dir / "mlp.pt",
    )

    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")

    print(f"Test RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")


if __name__ == "__main__":
    main()
