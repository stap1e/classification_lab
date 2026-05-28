"""CLI entry for running experiments from YAML profiles."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from config.load_config import list_profiles, load_experiment_config


def build_parser(*, default_profile: str) -> argparse.ArgumentParser:
    env_default = os.environ.get("CLS_EXPERIMENT", default_profile)
    parser = argparse.ArgumentParser(
        description="Run CPC classification experiment from a YAML profile.",
    )
    parser.add_argument(
        "--profile",
        "-p",
        default=env_default,
        help=f"Experiment profile name (default: {env_default!r}, env CLS_EXPERIMENT)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to experiments.yaml (default: config/experiments.yaml)",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profile names and exit",
    )
    return parser


def main(*, default_profile: str = "single") -> None:
    parser = build_parser(default_profile=default_profile)
    args = parser.parse_args()

    if args.list_profiles:
        for name in list_profiles(args.config):
            print(name)
        return

    try:
        config = load_experiment_config(args.profile, args.config)
    except (FileNotFoundError, KeyError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f"Profile: {args.profile}")
    print(f"  train: {config.train_path}")
    print(f"  test:  {config.test_path}")
    print(f"  use_nse={config.use_nse}, n_folds={config.n_folds}, classifier={config.classifier}")

    from pipeline.experiment import run_experiment

    run_experiment(config)


if __name__ == "__main__":
    main()
