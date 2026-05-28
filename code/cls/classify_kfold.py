"""
K-fold cross-validation on a single dataset (default 5 folds).

Run from code/cls:
    python classify_kfold.py
"""

from config.defaults import ExperimentConfig
from pipeline.experiment import run_experiment


def main() -> None:
    config = ExperimentConfig(
        train_path="./0721/0728data_delete.xlsx",
        test_path=None,
        results_base_dir="./results_0728_delete",
        lab_describe="cpc1-2=0_cpc3-5=1_kfold",
        use_nse=False,
        classifier="svm",
        n_folds=5,
        test_size=0.2,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
