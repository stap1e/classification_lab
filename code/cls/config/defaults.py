"""Shared defaults for CPC classification experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

# CPC 1–2 → 0, CPC 3–5 → 1 (good vs poor outcome)
CPC_LABEL_BINS: tuple[int, int, int] = (0, 2, 5)
CPC_LABELS: tuple[int, int] = (0, 1)

ID_COLUMNS: Sequence[str] = ("CTid", "name")
NSE_COLUMNS: Sequence[str] = ("nse极值", "nse极值差")

ClassifierName = Literal[
    "svm",
    "logistic",
    "gaussian_nb",
    "xgboost",
    "lightgbm",
    "catboost",
]


@dataclass
class ExperimentConfig:
    """Runtime settings for a single training/evaluation run."""

    train_path: Path | str
    test_path: Path | str | None = None
    results_base_dir: Path | str = "./results"
    lab_describe: str = "cpc1-2=0_cpc3-5=1"
    use_nse: bool = False
    classifier: ClassifierName = "svm"
    random_state: int = 42
    # K-fold mode: when test_path is None and n_folds > 1, split from train_path
    n_folds: int = 1
    fold_random_states: Sequence[int] = field(
        default_factory=lambda: (3, 13, 42, 87, 1307)
    )
    test_size: float = 0.2
    exclude_cpc5: bool = False
    cpc_bins: tuple[int, int, int] = CPC_LABEL_BINS
    cpc_labels: tuple[int, int] = CPC_LABELS

    def resolve_paths(self) -> None:
        self.train_path = Path(self.train_path)
        if self.test_path is not None:
            self.test_path = Path(self.test_path)
        self.results_base_dir = Path(self.results_base_dir)
