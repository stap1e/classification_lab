"""Data loading and CPC label preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config.defaults import CPC_LABEL_BINS, CPC_LABELS, ID_COLUMNS, NSE_COLUMNS


def load_excel(path: Path | str) -> pd.DataFrame:
    return pd.read_excel(Path(path))


def assign_cpc_labels(
    df: pd.DataFrame,
    *,
    bins: tuple[int, int, int] = CPC_LABEL_BINS,
    labels: tuple[int, int] = CPC_LABELS,
    exclude_cpc5: bool = False,
) -> pd.DataFrame:
    """Add binary ``label`` from ``CPC`` and drop the original column."""
    if "label" in df.columns and "CPC" not in df.columns:
        return df.copy()

    out = df.copy()
    if exclude_cpc5:
        out = out[out["CPC"] != 5]
    out["label"] = pd.cut(out["CPC"], bins=list(bins), labels=list(labels))
    return out.drop(columns=["CPC"])


def strip_id_columns(
    df: pd.DataFrame,
    *,
    id_columns: tuple[str, ...] = tuple(ID_COLUMNS),
    nse_columns: tuple[str, ...] = tuple(NSE_COLUMNS),
    keep_nse: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split feature frame and optional NSE side table."""
    drop_cols = list(id_columns)
    nse_df = None
    if keep_nse:
        present_nse = [c for c in nse_columns if c in df.columns]
        if present_nse:
            nse_df = df[present_nse].copy()
        drop_cols = list(id_columns) + list(nse_columns)
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return features, nse_df


def prepare_labeled_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    use_nse: bool = False,
    exclude_cpc5: bool = False,
    bins: tuple[int, int, int] = CPC_LABEL_BINS,
    labels: tuple[int, int] = CPC_LABELS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Label CPC (if needed), drop metadata columns, optionally extract NSE."""
    train_labeled = assign_cpc_labels(
        train_df, bins=bins, labels=labels, exclude_cpc5=exclude_cpc5
    )
    test_labeled = assign_cpc_labels(
        test_df, bins=bins, labels=labels, exclude_cpc5=exclude_cpc5
    )
    train_x, train_nse = strip_id_columns(train_labeled, keep_nse=use_nse)
    test_x, test_nse = strip_id_columns(test_labeled, keep_nse=use_nse)
    return train_x, test_x, train_nse, test_nse


def load_split_tables(
    train_path: Path | str,
    test_path: Path | str | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_column: str = "CPC",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load fixed train/test files or stratified-split a single table."""
    train_path = Path(train_path)
    if test_path is not None:
        return load_excel(train_path), load_excel(test_path)

    full = load_excel(train_path)
    train_df, test_df = train_test_split(
        full,
        test_size=test_size,
        stratify=full[stratify_column],
        random_state=random_state,
    )
    return train_df, test_df


def merge_nse_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_nse: pd.DataFrame,
    test_nse: pd.DataFrame,
    selected_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply LASSO-selected CT features and append NSE columns."""
    train_body = train_features[selected_features].copy()
    train_body["label"] = train_features["label"].values
    test_body = test_features[selected_features].copy()
    train_with_nse = pd.concat([train_body.drop(columns=["label"]), train_nse], axis=1)
    train_with_nse["label"] = train_body["label"].values
    test_with_nse = pd.concat([test_body, test_nse], axis=1)
    return train_with_nse, test_with_nse
