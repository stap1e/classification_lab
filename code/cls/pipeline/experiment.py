"""End-to-end training and evaluation for one experiment configuration."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.defaults import ExperimentConfig
from models.factory import build_classifier
from pipeline.data import (
    load_excel,
    load_split_tables,
    merge_nse_features,
    prepare_labeled_frames,
)
from utils.pre4data import if_same, lasso_dimension_reduction
from utils.util import calculate_metrics, get_next_result_folder, save_results

METRIC_KEYS = ("ACC", "Recall", "Specificity", "Precision", "NPV", "AUC")


def _balanced_sample_weights(y: pd.Series) -> np.ndarray:
    weights = np.ones(len(y))
    for label in (0, 1):
        mask = y == label
        if mask.sum():
            weights[mask] = len(y) / (2 * mask.sum())
    return weights


def _fit_classifier(clf, X_train, y_train, sample_weights) -> None:
    name = clf.__class__.__name__.lower()
    if "gaussiannb" in name or "logisticregression" in name:
        clf.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        clf.fit(X_train, y_train)


def _run_single_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    *,
    fold: int | None = None,
    save_roc_dir: str | None = None,
) -> tuple[dict[str, float], list[str], str]:
    train_x, test_x, train_nse, test_nse = prepare_labeled_frames(
        train_df,
        test_df,
        use_nse=config.use_nse,
        exclude_cpc5=config.exclude_cpc5,
        bins=config.cpc_bins,
        labels=config.cpc_labels,
    )

    y_test = test_x["label"]
    data_lasso, selected_features, _best_alpha = lasso_dimension_reduction(train_x)

    if config.use_nse and train_nse is not None and test_nse is not None:
        train_final, test_final = merge_nse_features(
            train_x, test_x, train_nse, test_nse, selected_features
        )
    else:
        train_final = data_lasso
        test_final = test_x.drop(columns=["label"])[selected_features]

    X_train = train_final.drop(columns=["label"])
    y_train = train_final["label"]

    if_same(X_train, test_final)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(test_final)

    n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = float(n0 / max(n1, 1))
    clf = build_classifier(
        config.classifier,
        scale_pos_weight=scale_pos_weight,
        class_priors=((y_train == 0).mean(), (y_train == 1).mean()),
        random_state=config.random_state,
    )

    _fit_classifier(
        clf, X_train_scaled, y_train, _balanced_sample_weights(y_train)
    )

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, -1]
    mode = clf.__class__.__name__

    acc, recall, spec, prec, npv, auc = calculate_metrics(
        y_test,
        y_pred,
        y_prob,
        save_roc_path=save_roc_dir,
        mode=mode,
        fold=fold,
    )

    metrics = {
        "ACC": acc,
        "Recall": recall,
        "Specificity": spec,
        "Precision": prec,
        "NPV": npv,
        "AUC": auc or 0.0,
    }
    log = (
        f"fold={fold}\n"
        f"train size={len(X_train)}, test size={len(test_final)}\n"
        f"selected features ({len(selected_features)}): {selected_features}\n"
    )
    return metrics, selected_features, log


def run_experiment(config: ExperimentConfig) -> str:
    """Execute experiment, write results, return results directory path."""
    config.resolve_paths()
    lab = (
        f"{config.lab_describe}_withnse"
        if config.use_nse
        else f"{config.lab_describe}_withoutnse"
    )

    result_folder = get_next_result_folder(base_path=str(config.results_base_dir))
    mode = build_classifier(config.classifier, scale_pos_weight=1.0).__class__.__name__
    ct_mode = config.train_path.stem
    save_path = os.path.join(
        result_folder,
        f"roc_curve_{mode}_time_{ct_mode}_{lab}_{datetime.now():%Y-%m-%d_%H-%M-%S}",
    )
    os.makedirs(save_path, exist_ok=True)

    all_metrics: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    feature_logs: list[str] = []
    split_logs: list[str] = []

    if config.n_folds > 1:
        # Split raw rows (keep CPC + NSE + CTid) so each fold can run full preprocessing.
        full_raw = load_excel(config.train_path)
        if config.exclude_cpc5:
            full_raw = full_raw[full_raw["CPC"] != 5].copy()

        for fold in range(1, config.n_folds + 1):
            print(f"============== Fold {fold} ==============")
            rs = config.fold_random_states[fold - 1]
            train_fold, test_fold = train_test_split(
                full_raw,
                test_size=config.test_size,
                stratify=full_raw["CPC"],
                shuffle=True,
                random_state=rs,
            )
            metrics, features, log = _run_single_fold(
                train_fold,
                test_fold,
                config,
                fold=fold,
                save_roc_dir=save_path,
            )
            for k in METRIC_KEYS:
                all_metrics[k].append(metrics[k])
            feature_logs.append(f"fold {fold}: {features}\n")
            split_logs.append(log)
    else:
        train_raw, test_raw = load_split_tables(
            config.train_path,
            config.test_path,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        metrics, features, log = _run_single_fold(
            train_raw, test_raw, config, save_roc_dir=save_path
        )
        for k in METRIC_KEYS:
            all_metrics[k].append(metrics[k])
        feature_logs.append(str(features))
        split_logs.append(log)

    results = (
        f"lab: {lab}\nclassifier: {mode}\n"
        f"random_state: {config.random_state}\nn_folds: {config.n_folds}\n\n"
    )
    results += "".join(split_logs)

    for k in METRIC_KEYS:
        vals = all_metrics[k]
        results += f"\n{k}: {np.mean(vals):.3f} ± {np.std(vals):.3f}\n"
        print(f"{k}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    for fl in feature_logs:
        results += fl

    probe = build_classifier(config.classifier, scale_pos_weight=1.0)
    for pm, val in probe.get_params().items():
        results += f"\nparameter {pm}: {val}\n"

    save_results(results, result_folder)
    return result_folder
