"""
Single-table experiment: stratified train/test split from one Excel file.

Run from code/cls:
    python classify_single.py
"""

from config.defaults import ExperimentConfig
from pipeline.experiment import run_experiment


def main() -> None:
    config = ExperimentConfig(
        train_path="./data/0804data_0.xlsx",
        test_path=None,
        results_base_dir="./results_0728_delete_new",
        lab_describe="cpc1-2=0_cpc3-5=1",
        use_nse=False,
        classifier="svm",
        random_state=42,
        test_size=0.2,
        n_folds=1,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
