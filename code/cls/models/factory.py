"""Classifier construction with shared hyperparameters."""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb

from config.defaults import ClassifierName


def build_classifier(
    name: ClassifierName,
    *,
    scale_pos_weight: float,
    class_priors: tuple[float, float] | None = None,
    random_state: int = 42,
):
    """Return a fresh sklearn-compatible classifier instance."""
    if name == "gaussian_nb":
        priors = list(class_priors) if class_priors else [0.5, 0.5]
        return GaussianNB(priors=priors)

    if name == "logistic":
        return LogisticRegression(random_state=random_state)

    if name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.016,
            max_depth=6,
            n_estimators=500,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
        )

    if name == "catboost":
        return CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.01,
            loss_function="Logloss",
            eval_metric="AUC",
            scale_pos_weight=scale_pos_weight,
            random_seed=random_state,
            verbose=0,
        )

    if name == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=["logloss"],
            learning_rate=0.016,
            max_depth=6,
            n_estimators=600,
            subsample=0.72,
            colsample_bytree=0.705,
            gamma=0.1,
            min_child_weight=1,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
        )

    if name == "svm":
        return SVC(
            kernel="rbf",
            C=10,
            gamma="auto",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )

    raise ValueError(f"Unknown classifier: {name!r}")
