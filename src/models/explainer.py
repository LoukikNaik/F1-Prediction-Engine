"""SHAP-based feature importance explanations."""

import numpy as np
import pandas as pd
import shap

from src.models.trainer import F1Ensemble
from src.pipeline.processors.feature_engineer import get_feature_columns
from src.utils.logger import logger


def compute_shap_values(
    model: F1Ensemble, df: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """Compute SHAP values for model explanations.

    Args:
        model: Trained F1Ensemble model.
        df: Feature-engineered DataFrame.

    Returns:
        Tuple of (shap_values array, feature_names list).
    """
    logger.info("Computing SHAP values...")
    feature_cols = get_feature_columns()
    X = df[feature_cols].values

    # Use TreeExplainer on LightGBM (primary model)
    explainer = shap.TreeExplainer(model.lgbm)
    shap_values = explainer.shap_values(X)

    # shap_values is a list of arrays (one per class), average across classes
    if isinstance(shap_values, list):
        shap_avg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_avg = np.abs(shap_values)

    logger.info("SHAP values computed")
    return shap_avg, feature_cols


def get_feature_importance(model: F1Ensemble, df: pd.DataFrame) -> pd.DataFrame:
    """Get global feature importance rankings.

    Returns:
        DataFrame with feature names and importance scores, sorted descending.
    """
    shap_values, feature_names = compute_shap_values(model, df)
    importance = shap_values.mean(axis=0)

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    result["importance_pct"] = result["importance"] / result["importance"].sum() * 100
    return result
