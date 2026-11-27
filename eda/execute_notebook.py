#!/usr/bin/env python3
"""Run ``eda/x5_eda.ipynb`` headlessly. The notebook is the only ``.ipynb`` in the EDA.

By default the notebook runs in memory and nothing is saved. With ``--in-place`` the cell outputs are written back into
``eda/x5_eda.ipynb`` so the notebook shows its results (the file stays the only ``.ipynb``). The results are also the
files written to the output directory (``summary.json``, ``tables/``, ``figures/``).

    python eda/execute_notebook.py --mode sample --input-dir data/x5/raw_sample
    python eda/execute_notebook.py --mode full --input-dir data/x5/extracted --in-place

Exit codes: 0 the notebook ran without errors; 1 a cell raised or the kernel failed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eda.x5_eda_lib import MODES  # noqa: E402

NOTEBOOK = REPO_ROOT / "eda" / "x5_eda.ipynb"


def run_notebook(
    mode: str = "sample",
    *,
    input_dir: str | Path | None = None,
    cutoff: str | None = None,
    output_dir: str | Path | None = None,
    timeout: int = 3600,
) -> nbformat.NotebookNode:
    """Execute the notebook and return it with its outputs. Errors are recorded as outputs, not raised."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    os.environ["X5_EDA_MODE"] = mode
    if input_dir:
        os.environ["X5_EDA_INPUT_DIR"] = str(Path(input_dir).resolve())
    if cutoff:
        os.environ["X5_CAMPAIGN_CUTOFF"] = cutoff
    if output_dir:
        os.environ["X5_EDA_OUTPUT_DIR"] = str(Path(output_dir).resolve())
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()
    return notebook


def cell_errors(notebook: nbformat.NotebookNode) -> list[dict]:
    return [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=MODES, default="sample")
    parser.add_argument("--input-dir")
    parser.add_argument("--cutoff")
    parser.add_argument(
        "--output-dir", help="where the notebook writes tables/figures (default eda/outputs/<mode>)"
    )
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per cell")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="save the cell outputs into eda/x5_eda.ipynb (it stays the only notebook)",
    )
    args = parser.parse_args(argv)
    try:
        notebook = run_notebook(
            args.mode,
            input_dir=args.input_dir,
            cutoff=args.cutoff,
            output_dir=args.output_dir,
            timeout=args.timeout,
        )
    except Exception as exc:  # kernel failures and timeouts
        print(f"notebook failed: {exc}", file=sys.stderr)
        return 1
    errors = cell_errors(notebook)
    if args.in_place:
        nbformat.write(notebook, NOTEBOOK)
        print(f"outputs saved into {NOTEBOOK}")
    if errors:
        print(f"notebook raised {len(errors)} error(s)", file=sys.stderr)
        return 1
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "eda" / "outputs" / args.mode
    print(f"notebook ({args.mode}) ran without errors; results in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
