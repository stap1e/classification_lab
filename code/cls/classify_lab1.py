"""
Lab1: fixed train/test split with optional NSE features.

Edit paths and flags below, then run from code/cls:
    python classify_lab1.py
"""

from config.defaults import ExperimentConfig
from pipeline.experiment import run_experiment


def main() -> None:
    config = ExperimentConfig(
        train_path="D:/thrid_beijing_hospital_data/0804lab1-train.xlsx",
        test_path="D:/thrid_beijing_hospital_data/0804lab1-test.xlsx",
        results_base_dir="D:/thrid_beijing_hospital_data/results_0804_lab1",
        lab_describe="cpc1-2=0_cpc3-5=1_lab1",
        use_nse=True,
        classifier="svm",
        random_state=42,
        n_folds=1,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
