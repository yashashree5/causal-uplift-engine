#!/usr/bin/env python3
"""Run the X5 EDA headlessly (same code the notebook uses) and write tables/figures/summary.json.

Examples::

    python eda/run_eda.py --mode sample --input-dir data/x5/raw_sample
    python eda/run_eda.py --mode full --input-dir data/x5/extracted

Precedence for the input directory: ``--input-dir`` > ``X5_EDA_INPUT_DIR`` > ``eda.input_dirs.<mode>`` in the YAML.
Outputs go to ``eda/outputs/<mode>/`` (git-ignored) unless ``--output-dir`` is given.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eda.x5_eda_lib import MODES, EdaError, EdaSettings, run_eda  # noqa: E402
from ingestion.x5_common import ConfigError, load_config  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
DEFAULT_EDA_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "x5" / "eda.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_EDA_CONFIG))
    parser.add_argument(
        "--mode", choices=MODES, default="sample", help="label of the run: sample or full data"
    )
    parser.add_argument("--input-dir", help="directory with the five <table>.csv(.gz) files")
    parser.add_argument("--output-dir", help="override eda.output_root/<mode>")
    parser.add_argument("--cutoff", help="campaign cutoff (YYYY-MM-DD[ HH:MM:SS]); overrides campaign.cutoff")
    parser.add_argument(
        "--force", action="store_true", help="rebuild the typed work tables even if unchanged"
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv: list[str] | None = None, env: dict[str, str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = dict(os.environ if env is None else env)
    try:
        config = load_config(args.config)
        settings = EdaSettings.from_config(
            config,
            mode=args.mode,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cutoff=args.cutoff,
            env=env,
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        results = run_eda(settings, force=args.force, figures=not args.no_figures)
    except EdaError as exc:
        print(f"EDA failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    run = results.sections["run"]
    print(f"EDA ({run['mode']}) finished in {run['total_seconds']}s; outputs in {settings.output_dir}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
