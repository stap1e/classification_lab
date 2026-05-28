#!/usr/bin/env python3
"""
Unified CLI for all experiment profiles.

Examples:
    python run_experiment.py --list-profiles
    python run_experiment.py -p lab1
    python run_experiment.py -p kfold_nse --config config/experiments.yaml
    CLS_DATA_ROOT=/path/to/data python run_experiment.py -p kfold_nse
"""

from config.cli import main

if __name__ == "__main__":
    main(default_profile="single")
