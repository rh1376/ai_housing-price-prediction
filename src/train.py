from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src.utils import resolve_repo_path


from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from src.utils import evaluate_model, load_arff_to_df, plot_actual_vs_predicted, plot_residuals, save_json, seed_everything


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


def get_models(seed: int) -> Dict[str, Tuple[object, Dict[str, object]]]:
    models: Dict[str, Tuple[object, Dict[str, object]]] = {
        "Ridge": (
            Ridge(random_state=seed),
            {
                "model__alpha": uniform(0.1, 50.0),
                "model__solver": ["auto", "lsqr", "sparse_cg", "sag"],
            },
        ),
        "DecisionTree": (
            DecisionTreeRegressor(random_state=seed),
            {
                "model__max_depth": randint(2, 30),
                "model__min_samples_split": randint(2, 20),
                "model__min_samples_leaf": randint(1, 10),
            },
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=seed, n_jobs=-1),
            {
                "model__n_estimators": randint(50, 300),
                "model__max_depth": randint(3, 30),
                "model__min_samples_split": randint(2, 20),
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", "log2", 1.0],
            },
        ),
    }

    try:
        import xgboost as xgb

        models["XGBoost"] = (
            xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=seed,
                n_estimators=300,
                n_jobs=-1,
            ),
            {
                "model__learning_rate": uniform(0.01, 0.3),
                "model__max_depth": randint(3, 10),
                "model__subsample": uniform(0.7, 0.25),        # [0.7, 0.95)
                "model__colsample_bytree": uniform(0.7, 0.25),
            },
        )
        print("XGBoost available. Including XGBoost in model search.")
    except Exception:
        print("XGBoost not available. Skipping XGBoost.")

    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate housing price models.")
    parser.add_argument("--data", required=True, help="Path to dataset.arff")
    parser.add_argument("--out_dir", default=".", help="Output directory for artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    data_path = resolve_repo_path(args.data)
    out_dir = resolve_repo_path(args.out_dir)
    models_dir = out_dir / "models"
    reports_dir = out_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = load_arff_to_df(data_path)

    print("Preparing train/test split...")
    df = df.drop(columns=["id"], errors="ignore")
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )

    preprocessor = build_preprocessor()
    models = get_models(args.seed)

    results = []
    best_model_name = None
    best_estimator = None
    best_rmse = np.inf
    best_params = {}

    for name, (estimator, param_dist) in models.items():
        print(f"Training {name} with RandomizedSearchCV...")
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=args.seed,
        )
        search.fit(X_train, y_train)

        y_pred = search.best_estimator_.predict(X_test)
        rmse, mae, r2 = evaluate_model(y_test.to_numpy(), y_pred)
        results.append(
            {
                "model": name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )
        print(f"{name} test RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_estimator = search.best_estimator_
            best_params = search.best_params_

    if best_estimator is None:
        raise RuntimeError("No models were trained successfully.")

    print(f"Best model by test RMSE: {best_model_name} ({best_rmse:.4f})")
    joblib.dump(best_estimator, models_dir / "best_model.joblib")

    metrics_df = pd.DataFrame(results).sort_values("rmse")
    metrics_df.to_csv(reports_dir / "metrics.csv", index=False)

    save_json(best_params, reports_dir / "best_params.json")

    best_predictions = best_estimator.predict(X_test)
    plot_actual_vs_predicted(y_test.to_numpy(), best_predictions, reports_dir / "actual_vs_predicted.png")
    plot_residuals(y_test.to_numpy(), best_predictions, reports_dir / "residuals.png")

    print(f"Artifacts saved to {models_dir} and {reports_dir}.")


# Run the script only when executed directly (not when imported)
if __name__ == "__main__":
    main()
