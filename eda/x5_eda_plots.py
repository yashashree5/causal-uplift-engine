"""Figures for the X5 EDA, drawn only from the tables in :class:`eda.x5_eda_lib.Results`."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, the notebook displays them from disk

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from eda.x5_eda_lib import Results  # noqa: E402


def _bars(ax, frame: pd.DataFrame, *, xlabel: str, log_y: bool = False) -> None:
    if frame.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    ax.bar(frame["bin_start"], frame["n"], width=frame["bin_width"].iloc[0], align="edge", color="#4c72b0")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if log_y:
        ax.set_yscale("log")


def _hist_figure(
    results: Results, table: str, xlabel: str, title: str, out: Path, *, log_y: bool = False
) -> Path | None:
    frame = results.tables.get(table)
    if frame is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    _bars(ax, frame, xlabel=xlabel, log_y=log_y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _uplift_figure(results: Results, kind: str, out: Path) -> Path | None:
    table = results.tables.get("uplift_by_segment")
    if table is None:
        return None
    frame = table[table["segment_kind"] == kind].reset_index(drop=True)
    if frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = frame["uplift"].astype(float)
    err_low = (y - frame["ci_low"].astype(float)).clip(lower=0)
    err_high = (frame["ci_high"].astype(float) - y).clip(lower=0)
    colors = ["#c44e52" if bool(s) else "#4c72b0" for s in frame["sparse"].astype(str).str.lower().eq("true")]
    ax.bar(range(len(frame)), y, yerr=[err_low, err_high], capsize=3, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels(frame["segment"], rotation=30, ha="right")
    ax.set_ylabel("uplift (treated - control)")
    ax.set_title(f"Uplift by {kind} (95% CI; red = sparse segment)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _balance_figure(results: Results, out: Path) -> Path | None:
    frame = results.tables.get("covariate_balance")
    if frame is None or frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(frame["covariate"], frame["smd"].astype(float), color="#55a868")
    ax.axvline(0, color="black", linewidth=0.8)
    for edge in (-0.1, 0.1):
        ax.axvline(edge, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("standardised mean difference (treated - control)")
    ax.set_title("Covariate balance between arms")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _monthly_figure(results: Results, out: Path) -> Path | None:
    frame = results.tables.get("transactions_by_week")
    if frame is None or frame.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(pd.to_datetime(frame["week_start"]), frame["n_transactions"].astype(float), marker="o")
    ax.set_ylabel("transactions per week")
    ax.set_title("Transactions over time (weeks starting on date)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def make_figures(results: Results, figures_dir: Path) -> list[Path]:
    """Write all figures; skip a figure when its source table is missing. Returns the written paths."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        _hist_figure(
            results,
            "hist_spend_per_transaction",
            "purchase_sum per transaction",
            "Spend per transaction",
            figures_dir / "spend_per_transaction.png",
            log_y=True,
        ),
        _hist_figure(
            results,
            "hist_transactions_per_client",
            "transactions",
            "Transactions per client",
            figures_dir / "transactions_per_client.png",
            log_y=True,
        ),
        _hist_figure(results, "hist_age", "age (valid range)", "Client age", figures_dir / "age.png"),
        _hist_figure(
            results,
            "hist_recency_days",
            "days since last purchase",
            "Recency (vs anchor)",
            figures_dir / "recency_days.png",
        ),
        _hist_figure(
            results,
            "hist_stores_per_client",
            "distinct stores",
            "Stores per client",
            figures_dir / "stores_per_client.png",
            log_y=True,
        ),
        _monthly_figure(results, figures_dir / "transactions_over_time.png"),
        _balance_figure(results, figures_dir / "covariate_balance.png"),
        *[
            _uplift_figure(results, kind, figures_dir / f"uplift_by_{kind}.png")
            for kind in ("age_band", "spend_tier", "recency_band", "gender")
        ],
    ]
    return [p for p in jobs if p is not None]
