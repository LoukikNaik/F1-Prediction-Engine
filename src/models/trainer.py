"""Train ensemble ML models for F1 race prediction."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config.settings import MODELS_DIR, RANDOM_SEED
from src.pipeline.processors.feature_engineer import get_feature_columns
from src.utils.constants import GRID_SIZE
from src.utils.logger import logger


class F1Ensemble:
    """Ensemble model combining LightGBM, XGBoost, and Logistic Regression."""

    def __init__(self):
        self.lgbm = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            objective="multiclass",
            num_class=GRID_SIZE,
            random_state=RANDOM_SEED,
            verbose=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        self.xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=GRID_SIZE,
            random_state=RANDOM_SEED,
            verbosity=0,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        self.lr = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
        self.scaler = StandardScaler()
        self.feature_columns = get_feature_columns()
        self.weights = [0.45, 0.35, 0.20]  # LightGBM, XGBoost, LogReg
        self.is_trained = False

    def _prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Convert position to 0-indexed class label."""
        return (df["position_filled"].clip(1, GRID_SIZE) - 1).astype(int).values

    def train(self, train_df: pd.DataFrame) -> dict:
        """Train all models in the ensemble.

        Args:
            train_df: Feature-engineered DataFrame.

        Returns:
            Dict with training metrics.
        """
        logger.info("Training ensemble models...")

        X = train_df[self.feature_columns].values
        y = self._prepare_target(train_df)
        sample_weights = train_df["era_weight"].values

        # Scale features for logistic regression
        X_scaled = self.scaler.fit_transform(X)

        # Train LightGBM
        logger.info("Training LightGBM...")
        self.lgbm.fit(X, y, sample_weight=sample_weights)

        # Train XGBoost
        logger.info("Training XGBoost...")
        self.xgb.fit(X, y, sample_weight=sample_weights)

        # Train Logistic Regression (with calibration)
        logger.info("Training Logistic Regression...")
        self.lr.fit(X_scaled, y, sample_weight=sample_weights)

        self.is_trained = True

        # Compute training accuracy
        preds = self.predict_proba(train_df)
        pred_positions = preds.argmax(axis=1) + 1
        actual = train_df["position_filled"].values
        accuracy = (pred_positions == actual).mean()
        top3_acc = np.mean([
            actual[i] in np.argsort(preds[i])[-3:] + 1
            for i in range(len(actual))
        ])

        metrics = {
            "train_accuracy": float(accuracy),
            "train_top3_accuracy": float(top3_acc),
            "n_samples": len(train_df),
        }
        logger.info(f"Training complete. Accuracy: {accuracy:.3f}, Top-3: {top3_acc:.3f}")
        return metrics

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions for each position.

        Args:
            df: Feature-engineered DataFrame.

        Returns:
            Array of shape (n_drivers, 20) with position probabilities.
        """
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        proba_lgbm = self.lgbm.predict_proba(X)
        proba_xgb = self.xgb.predict_proba(X)
        proba_lr = self.lr.predict_proba(X)

        # Ensure all have same number of classes
        n_classes = GRID_SIZE
        for proba in [proba_lgbm, proba_xgb, proba_lr]:
            if proba.shape[1] < n_classes:
                padding = np.zeros((proba.shape[0], n_classes - proba.shape[1]))
                proba = np.hstack([proba, padding])

        # Weighted ensemble
        ensemble_proba = (
            self.weights[0] * proba_lgbm[:, :n_classes]
            + self.weights[1] * proba_xgb[:, :n_classes]
            + self.weights[2] * proba_lr[:, :n_classes]
        )

        # Normalize rows to sum to 1
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        ensemble_proba = ensemble_proba / np.maximum(row_sums, 1e-10)

        return ensemble_proba

    def save(self, path: Path | None = None) -> Path:
        """Save trained models to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "ensemble_model.pkl"

        with open(path, "wb") as f:
            pickle.dump({
                "lgbm": self.lgbm,
                "xgb": self.xgb,
                "lr": self.lr,
                "scaler": self.scaler,
                "weights": self.weights,
                "feature_columns": self.feature_columns,
            }, f)
        logger.info(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "F1Ensemble":
        """Load trained models from disk."""
        if path is None:
            path = MODELS_DIR / "ensemble_model.pkl"

        with open(path, "rb") as f:
            data = pickle.load(f)

        model = cls()
        model.lgbm = data["lgbm"]
        model.xgb = data["xgb"]
        model.lr = data["lr"]
        model.scaler = data["scaler"]
        model.weights = data["weights"]
        model.feature_columns = data["feature_columns"]
        model.is_trained = True
        logger.info(f"Model loaded from {path}")
        return model
