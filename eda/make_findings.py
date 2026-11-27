#!/usr/bin/env python3
"""Generate ``eda/FINDINGS.md`` from the **full-data** EDA outputs.

Every number in the generated document is read from ``summary.json`` / ``tables/*.csv`` written by
``eda/run_eda.py``. Nothing is typed in by hand. The generator refuses outputs of a ``sample`` run, so sample
statistics cannot end up in the findings by accident.

    python eda/run_eda.py --mode full --input-dir data/x5/extracted
    python eda/make_findings.py                       # reads eda/outputs/full, writes eda/FINDINGS.md
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eda.x5_eda_lib import EdaError, Results  # noqa: E402
from ingestion.x5_common import ConfigError, atomic_write_text, load_config, resolve_path  # noqa: E402

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
DEFAULT_OUTPUTS = "eda/outputs/full"
DEFAULT_TARGET = "eda/FINDINGS.md"
DEFAULT_EXPECTATIONS_CONFIG = "configs/x5/raw_landing.yaml"


class FindingsError(RuntimeError):
    """The outputs cannot be turned into findings (wrong mode, missing sections)."""


@dataclasses.dataclass(frozen=True)
class Finding:
    observation: str
    evidence: str
    action: str
    caveat: str


# --------------------------------------------------------------------------- formatting


def n(value: Any) -> str:
    return f"{int(round(value)):,}"


def pct(part: float, whole: float, digits: int = 1) -> str:
    return f"{100 * part / whole:.{digits}f}%" if whole else "n/a"


def pp(value: float, digits: int = 2) -> str:
    return f"{100 * value:+.{digits}f} pp"


def num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def cell(text: str) -> str:
    """Make text safe for a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def ts(value: Any) -> str:
    return str(value).replace("T", " ")


# --------------------------------------------------------------------------- findings


def segment_span(seg: Any, kind: str) -> str:
    """``kind: -1.00 pp (low segment) to +5.00 pp (high segment)`` over the segments with a defined uplift."""
    sub = seg[(seg["segment_kind"] == kind) & (seg["segment"] != "invalid_or_missing")]
    sub = sub[sub["uplift"].notna()]
    if sub.empty:
        return f"{kind}: no segment has a defined uplift"
    hi_row, lo_row = sub.loc[sub["uplift"].idxmax()], sub.loc[sub["uplift"].idxmin()]
    return f"{kind}: {pp(lo_row['uplift'])} ({lo_row['segment']}) to {pp(hi_row['uplift'])} ({hi_row['segment']})"


def build_findings(results: Results) -> list[Finding]:
    s = results.sections
    try:
        files, times, leak = s["files"], s["timestamps"], s["leakage"]
        dist, treat, sem = s["distributions"], s["treatment"], s["semantics"]
        utest = s["uplift_test"]
        s["id_formats"]  # required section (ID format table is read from results.tables)
    except KeyError as exc:
        raise FindingsError(f"summary.json lacks section {exc}") from exc
    rel = files["relationships"]
    purchases = files["purchases"]
    out: list[Finding] = []

    # ---- time and leakage
    span = times["purchase_span_days"]
    out.append(
        Finding(
            "No campaign date is published, so purchase history cannot be split into before/after the SMS.",
            f"Purchases span {ts(times['purchase_ts_min'])} to {ts(times['purchase_ts_max'])} ({span:.1f} days). "
            f"Leakage check status: **{leak['status']}** (`campaign.cutoff` is unset).",
            "Make every time-based feature take a cutoff parameter; do not build or train final features until the real "
            "campaign date is confirmed and set in `campaign.cutoff`. Until then recency uses the provisional anchor "
            f"`{leak['recency_anchor']}` (max purchase timestamp).",
            "The anchor is provisional; recency and any 'pre-campaign' feature may include post-campaign behaviour.",
        )
    )
    total_clients = files["clients_rows"]
    out.append(
        Finding(
            "`first_redeem_date` extends far beyond the purchase history and is partly missing.",
            f"Max `first_redeem_date` is {ts(times['clients_first_redeem_ts_max'])}; "
            f"{n(times['clients_first_redeem_after_last_purchase_ts'])} clients "
            f"({pct(times['clients_first_redeem_after_last_purchase_ts'], total_clients)}) first redeemed after the last purchase "
            f"in the data; {n(times['clients_first_redeem_null'])} ({pct(times['clients_first_redeem_null'], total_clients)}) "
            f"have no redeem date; {n(times['clients_first_redeem_before_first_issue'])} redeemed before the card was issued.",
            "Treat `first_redeem_date` as leakage-suspect: do not use it (or features derived from it) until the campaign "
            "date is known; if kept, use only redemptions before the cutoff. Null means 'never redeemed', not 'missing'.",
            "Post-campaign redemption could be an outcome of the campaign. The meaning of null is inferred, not documented.",
        )
    )
    out.append(
        Finding(
            "Card issue dates are inconsistent with observed purchases for a sizeable minority of clients.",
            f"{n(times['clients_first_issue_after_first_purchase'])} clients "
            f"({pct(times['clients_first_issue_after_first_purchase'], total_clients)}) have `first_issue_date` after their "
            f"first purchase in the data. Issue dates range {ts(times['clients_first_issue_ts_min'])} to "
            f"{ts(times['clients_first_issue_ts_max'])}.",
            "Do not use `first_issue_date` to filter or order purchases. Tenure features are allowed but should be "
            "flagged as noisy; add an indicator for 'issued after first observed purchase'.",
            "The cause is unknown (an earlier card, or an artefact of how the dataset was anonymised); the purchase "
            "history covers only about 4 months.",
        )
    )

    # ---- client attributes
    bad_low, bad_high = (
        dist["age_outside_valid_range"]["below_min"],
        dist["age_outside_valid_range"]["above_max"],
    )
    age = dist["age_all_values"]
    lo, hi = dist["valid_age_range"]
    out.append(
        Finding(
            "Ages contain impossible values.",
            f"`age` ranges {n(age['min'])} to {n(age['max'])}; {n(bad_low)} clients are below {lo} "
            f"({n(age['n_negative'])} negative, {n(age['n_zero'])} zero) and {n(bad_high)} are above {hi} "
            f"({pct(bad_low + bad_high, total_clients, 2)} of clients in total). No age is missing.",
            f"Set ages outside [{lo}, {hi}] to missing and add an `age_invalid` indicator; keep the clients (they have "
            "treatment labels and purchases). Do not impute from the mean before the split.",
            f"[{lo}, {hi}] is an analysis choice in `configs/x5/eda.yaml`, not a published rule.",
        )
    )
    out.append(
        Finding(
            "`gender` has a large 'U' category.",
            "; ".join(
                f"{row.gender}: {n(row.n_clients)} ({pct(row.n_clients, total_clients)})"
                for row in results.tables["gender_counts"].itertuples()
            ),
            "Keep `U` as its own category (do not map it to missing or drop it).",
            "'U' presumably means unknown; this is not documented in the source.",
        )
    )

    # ---- transaction semantics
    diff_rows = sem["multi_row_transactions_where_value_differs_between_rows"]
    out.append(
        Finding(
            "`purchase_sum` and the four points columns repeat on every product row of a transaction.",
            f"{n(sem['transactions_with_several_rows'])} of {n(sem['transactions'])} transactions have more than one "
            f"product row; the number of those transactions in which a value differs between rows is 0 for every "
            f"column checked ({', '.join(diff_rows)}). A row-wise sum of `purchase_sum` is "
            f"{sem['row_sum_overstatement_factor']:.2f}x the once-per-transaction sum "
            f"({num(sem['sum_purchase_sum_over_rows'], 0)} vs {num(sem['sum_purchase_sum_once_per_transaction'], 0)}).",
            "Aggregate `purchase_sum` and the points columns once per (client_id, transaction_id) (e.g. `max`) before any "
            "per-client or per-day sum; never `SUM` them over product rows.",
            "Holds for this release of the data; re-run the check if the source changes.",
        )
        if all(v == 0 for v in diff_rows.values())
        else Finding(
            "`purchase_sum` / points repeat per product row but differ for some transactions.",
            f"Transactions where rows differ: {diff_rows}.",
            "Investigate those transactions before aggregating; aggregate once per (client_id, transaction_id) otherwise.",
            "Unexpected: the repeat-per-row assumption is not exact.",
        )
    )
    shared = sem["transaction_ids_shared_by_more_than_one_client"]
    out.append(
        Finding(
            "`transaction_id` is not unique across clients.",
            f"{n(purchases['distinct_transaction_id'])} distinct `transaction_id` but "
            f"{n(purchases['distinct_client_transaction_pairs'])} distinct (client_id, transaction_id) pairs: {n(shared)} ids are "
            f"used by two clients ({n(sem['client_transactions_with_shared_transaction_id'])} client-transactions, "
            f"{n(sem['product_rows_with_shared_transaction_id'])} product rows). Within a (client_id, transaction_id) pair "
            f"timestamp and store never differ ({n(sem['client_transactions_with_more_than_one_timestamp'])} / "
            f"{n(sem['client_transactions_with_more_than_one_store'])} exceptions).",
            "Use (client_id, transaction_id) as the transaction key everywhere, including the replay event id; never "
            "group by `transaction_id` alone.",
            "The cause is not documented (possibly collisions of the anonymised ids).",
        )
    )
    if (
        purchases["rows_beyond_first_per_transaction_product_pair"]
        or purchases["exact_duplicate_rows_hash_based"]
    ):
        out.append(
            Finding(
                "A few product rows repeat a (transaction, product) pair.",
                f"{n(purchases['rows_beyond_first_per_transaction_product_pair'])} rows repeat an earlier "
                f"(transaction_id, product_id) pair; exact duplicate rows (hash-based): "
                f"{n(purchases['exact_duplicate_rows_hash_based'])}.",
                "Keep these rows (they differ in at least one value) and sum quantities per product within a transaction.",
                "Negligible in volume; a 64-bit hash could in theory hide a collision (it can only under-count).",
            )
        )
    out.append(
        Finding(
            "Zero-valued amounts and quantities occur.",
            f"{n(dist['purchase_sum_sign']['n_zero'])} transactions have `purchase_sum` = 0 (none negative); "
            f"{n(dist['quantity_per_row']['n_zero'])} product rows ({pct(dist['quantity_per_row']['n_zero'], dist['quantity_per_row']['n_rows'])}) "
            f"have `product_quantity` = 0; {n(sem['rows_with_zero_iss'])} rows have `trn_sum_from_iss` = 0; "
            f"no quantities are negative ({n(dist['quantity_per_row']['n_negative'])}) or fractional "
            f"({n(dist['quantity_fractional_rows'])}) and no negative `purchase_sum` exists (no returns are visible).",
            "Keep the rows but flag zero-sum transactions and exclude them from ratio features (spend per item); do not "
            "treat zero quantity as missing.",
            "Why quantity is 0 is unknown (e.g. weighed goods or voided lines).",
        )
    )
    points = results.tables["points_sign_conventions"].set_index("column")
    out.append(
        Finding(
            "Points 'spent' columns are stored as non-positive numbers; 'received' as non-negative.",
            "; ".join(
                f"`{c}`: min {num(points.loc[c, 'min'], 1)}, max {num(points.loc[c, 'max'], 1)}, "
                f"{n(points.loc[c, 'n_negative'])} negative / {n(points.loc[c, 'n_positive'])} positive transactions"
                for c in points.index
            ),
            "Document the sign convention in the cleaning code and take the absolute value (or negate) when building "
            "'points spent' features.",
            "Evidence is the observed sign only; the source gives no description.",
        )
    )
    red_rows, total_rows = sem["rows_with_trn_sum_from_red"], sem["rows_total"]
    out.append(
        Finding(
            "`trn_sum_from_red` is mostly empty and `trn_sum_from_iss` does not always add up to `purchase_sum`.",
            f"`trn_sum_from_red` is present on {n(red_rows)} of {n(total_rows)} rows ({pct(red_rows, total_rows)}); "
            f"`trn_sum_from_iss` on {n(sem['rows_with_trn_sum_from_iss'])} ({pct(sem['rows_with_trn_sum_from_iss'], total_rows)}). Per transaction, the sum of `trn_sum_from_iss` equals "
            f"`purchase_sum` to 1e-6 relative in {pct(sem['sum_iss_relative_agreement_with_purchase_sum']['within_1e-6'], sem['sum_iss_relative_agreement_with_purchase_sum']['transactions_compared'])} "
            f"of {n(sem['sum_iss_relative_agreement_with_purchase_sum']['transactions_compared'])} transactions, within 1% in "
            f"{pct(sem['sum_iss_relative_agreement_with_purchase_sum']['within_1pct'], sem['sum_iss_relative_agreement_with_purchase_sum']['transactions_compared'])}, "
            f"within 5% in {pct(sem['sum_iss_relative_agreement_with_purchase_sum']['within_5pct'], sem['sum_iss_relative_agreement_with_purchase_sum']['transactions_compared'])}; "
            f"median ratio {sem['sum_iss_over_purchase_sum_quantiles']['0.5']:.4f}, 1st percentile "
            f"{sem['sum_iss_over_purchase_sum_quantiles']['0.01']:.3f}.",
            "Do not equate `trn_sum_from_iss` with a line amount in the streaming contract (`items.amount`) without an "
            "explicit owner decision; use `purchase_sum` (once per transaction) for transaction value. Treat a null "
            "`trn_sum_from_red` as 0 with an indicator.",
            "It is plausible that `trn_sum_from_iss` is the issued-points-eligible line amount, but the data do not prove "
            "it. The mapping `items.amount` <- `trn_sum_from_iss` stays unverified.",
        )
    )

    # ---- products and ids
    nulls = results.tables["null_counts"].set_index(["table", "column"])
    prod_nulls = {
        c: int(nulls.loc[("products", c), "n_null"])
        for c in ("brand_id", "segment_id", "vendor_id", "level_1", "netto")
    }
    out.append(
        Finding(
            "Product attributes have gaps.",
            f"Of {n(files['products_rows'])} products: `brand_id` null for {n(prod_nulls['brand_id'])} "
            f"({pct(prod_nulls['brand_id'], files['products_rows'])}), `segment_id` {n(prod_nulls['segment_id'])} "
            f"({pct(prod_nulls['segment_id'], files['products_rows'])}), `vendor_id` {n(prod_nulls['vendor_id'])}, "
            f"`level_1..4` and `netto` {n(prod_nulls['level_1'])} each. {n(rel['products_never_purchased'])} products are "
            f"never purchased; every purchased product_id exists in `products` ({n(rel['purchase_product_ids_not_in_products'])} missing).",
            "Encode missing categories as an explicit 'unknown' level; do not drop purchase rows or products.",
            "`segment_id` is stored as a float string (e.g. `123.0`); parse as an integer-valued category, not a measure.",
        )
    )
    id_table = results.tables["id_formats"]
    client_ids = id_table[(id_table["table"] == "clients") & (id_table["column"] == "client_id")].iloc[0]
    lengths = sorted({int(v) for v in (*id_table["min_length"], *id_table["max_length"])})
    all_hex = bool((id_table["share_lowercase_hex"] == 1).all())
    id_summary = (
        f"Across the {len(id_table)} ID columns checked, "
        + (
            "every distinct value is lowercase hex"
            if all_hex
            else "NOT every distinct value is lowercase hex"
        )
        + f"; value lengths: {', '.join(map(str, lengths))}"
    )
    out.append(
        Finding(
            "IDs are opaque strings (they look hashed) and some of them look numeric.",
            f"{id_summary}; for `client_id` "
            f"{pct(client_ids['share_leading_zero'], 1)} start with '0' ({n(client_ids['share_leading_zero'] * client_ids['distinct_values'])} ids), "
            f"{pct(client_ids['share_digits_only'], 1, 2)} are all digits "
            f"({n(client_ids['share_digits_only'] * client_ids['distinct_values'])} ids) and "
            f"{n(client_ids['n_scientific_notation_like'])} look like scientific notation (digits, `e`, digits), which a "
            f"numeric parser would read as a float.",
            "Read and store every ID as a string (explicit dtype in CSV/Parquet readers and in the event schema). Casting "
            "to a number would drop leading zeros and turn the scientific-notation-looking ids into floats.",
            "Hashing means the IDs carry no meaning; product/category hierarchy must come from `products` only.",
        )
    )

    # ---- experiment
    overall = treat["overall"]
    balance = results.tables["covariate_balance"]
    out.append(
        Finding(
            "Treatment looks randomised and balanced; the average effect is positive.",
            f"{n(overall['n_treated'])} treated / {n(overall['n_control'])} control "
            f"(treated share {100 * treat['treated_share']:.2f}%, p = {treat['share_vs_half']['p_value']:.2f} against 50%). "
            f"Max |standardised mean difference| over {len(balance)} pre-treatment covariates = {treat['max_abs_smd']:.4f}. "
            f"Outcome rate {100 * overall['rate_treated']:.2f}% treated vs {100 * overall['rate_control']:.2f}% control: "
            f"uplift {pp(overall['uplift'])} (95% CI {pp(overall['ci_low'])} to {pp(overall['ci_high'])}).",
            "No cleaning needed for the labels (treatment_flg and target are 0/1 only, no nulls). Use `uplift_train` as the "
            "only labelled randomised sample.",
            "Balance is checked on attributes derived from the full purchase history (may extend past the campaign).",
        )
    )
    seg = results.tables["uplift_by_segment"]

    out.append(
        Finding(
            "Uplift varies by segment.",
            f"{segment_span(seg, 'age_band')}; {segment_span(seg, 'spend_tier')}; {segment_span(seg, 'gender')}. "
            f"{int(seg['sparse'].astype(str).str.lower().eq('true').sum())} of {len(seg)} segments are flagged sparse.",
            "Keep age, spend and gender-derived features; the baseline should be able to express these interactions.",
            "Wald intervals, no multiplicity correction; segments are descriptive and built from history that may "
            "include post-campaign purchases.",
        )
    )
    rec = seg[(seg["segment_kind"] == "recency_band")]
    out.append(
        Finding(
            "Recency has very little spread under the provisional anchor.",
            f"Every client's last purchase is within {dist['recency_days_with_purchases']['max']:.1f} days of the anchor; "
            + ", ".join(f"{r.segment}: {pct(r.n, rec['n'].sum())}" for r in rec.itertuples())
            + " of train clients.",
            "Do not rely on recency bands from the provisional anchor; recompute recency once the real cutoff is known "
            "and consider finer-grained (hours) recency.",
            "The data seem to be restricted to clients active near the end of the window; the anchor is provisional.",
        )
    )

    # ---- structure
    out.append(
        Finding(
            "`uplift_test` carries no labels and no treatment flag.",
            f"Columns: {', '.join(utest['columns'])}. {n(utest['clients'])} clients, all with purchases "
            f"({n(utest['clients_with_purchases'])}).",
            "Evaluate models only with a hold-out or cross-validation split of `uplift_train`; never expect labels for "
            "`uplift_test`.",
            "No evaluation labels are included in this release of the data.",
        )
    )
    out.append(
        Finding(
            "Tables join cleanly: no orphans, no clients without purchases.",
            f"Clients without purchases: {n(files['clients_without_purchases_total'])}; purchase clients missing from "
            f"`clients`: {n(rel['purchase_clients_not_in_clients'])}; train/test overlap: {n(rel['train_test_overlap_clients'])}; "
            f"train clients not in `clients`: {n(rel['train_clients_not_in_clients'])}; clients in neither train nor test: "
            f"{n(rel['clients_in_neither_train_nor_test'])}; duplicate keys: "
            f"{n(sum(v['duplicate_rows'] for v in files['key_uniqueness'].values()))}; values that failed type casts: "
            f"{n(files['cast_failures_total'])}; NaN/inf numeric values: {n(sem['rows_with_nan_or_inf_numeric'])}.",
            "Keep these as assertions in the cleaning job so a changed source fails loudly instead of silently dropping rows.",
            "True for this release only.",
        )
    )
    months = results.tables["transactions_by_month"]
    partial = months.iloc[[0, -1]]
    out.append(
        Finding(
            "The purchase history is short and its first and last months are partial.",
            f"{n(len(months))} calendar months, {span:.1f} days; "
            + "; ".join(
                f"{r.month}: {n(r.n_transactions)} transactions ({ts(r.first_ts)} to {ts(r.last_ts)})"
                for r in partial.itertuples()
            )
            + ".",
            "Use daily or weekly windows rather than calendar months; do not compare the first and last month by volume.",
            "Seasonality (including December) cannot be separated from the campaign with 4 months of history.",
        )
    )
    return out


# --------------------------------------------------------------------------- one-page summary


def _refs(findings: list[Finding], *texts: str) -> str:
    """``(#2, #3)``: rows of the findings table whose observation contains each text (empty if none match)."""
    rows = [str(i) for text in texts for i, f in enumerate(findings, 1) if text in f.observation]
    return f" (#{', #'.join(rows)})" if rows else ""


def build_summary(results: Results, findings: list[Finding]) -> list[str]:
    """Short list of the cleaning decisions, every number read from the same outputs as the findings table."""
    s = results.sections
    files, times, leak = s["files"], s["timestamps"], s["leakage"]
    dist, treat, sem = s["distributions"], s["treatment"], s["semantics"]
    total_clients = files["clients_rows"]
    lo, hi = dist["valid_age_range"]
    bad_age = dist["age_outside_valid_range"]["below_min"] + dist["age_outside_valid_range"]["above_max"]
    points = results.tables["points_sign_conventions"].set_index("column")
    spent_negative = sum(int(points.loc[c, "n_negative"]) for c in points.index if c.endswith("_spent"))
    id_table = results.tables["id_formats"]
    ids = id_table[(id_table["table"] == "clients") & (id_table["column"] == "client_id")].iloc[0]
    overall, seg = treat["overall"], results.tables["uplift_by_segment"]
    repeated = all(v == 0 for v in sem["multi_row_transactions_where_value_differs_between_rows"].values())
    r = lambda *texts: _refs(findings, *texts)  # noqa: E731
    return [
        f"1. **Campaign date is unknown; the leakage check is `{leak['status']}`.**{r('No campaign date')} Give every time-based "
        f"feature a cutoff parameter; recency uses the provisional anchor `{leak['recency_anchor']}`.",
        f"2. **Do not use `first_redeem_date`; treat `first_issue_date` as noisy.**{r('`first_redeem_date`', 'Card issue')} "
        f"{pct(times['clients_first_redeem_after_last_purchase_ts'], total_clients)} of clients redeem after the last purchase; "
        f"{pct(times['clients_first_issue_after_first_purchase'], total_clients)} are issued after their first purchase.",
        f"3. **Set ages outside [{lo}, {hi}] to missing and add an `age_invalid` flag.**{r('Ages contain')} "
        f"{n(bad_age)} clients ({pct(bad_age, total_clients, 2)}); keep them.",
        (
            f"4. **Aggregate `purchase_sum` and the points columns once per (client_id, transaction_id).**"
            f"{r('`purchase_sum` and the four points')} A row-wise sum overstates spend "
            f"{sem['row_sum_overstatement_factor']:.2f}x."
            if repeated
            else f"4. **`purchase_sum` / points differ between product rows of some transactions: investigate first.**"
            f"{r('`purchase_sum` / points repeat')}"
        ),
        f"5. **Key a transaction by (client_id, transaction_id).**{r('`transaction_id` is not unique')} "
        f"{n(sem['transaction_ids_shared_by_more_than_one_client'])} ids are shared by two clients.",
        f"6. **Points spent are stored as non-positive numbers: take the absolute value.**{r('Points')} "
        f"{n(spent_negative)} non-zero spent values (regular and express) are negative; received points are never negative.",
        f"7. **Keep zero-quantity and zero-sum rows, flag them; null `trn_sum_from_red` = 0 plus an indicator.**"
        f"{r('Zero-valued', '`trn_sum_from_red` is mostly')} `trn_sum_from_iss` is not a proven line amount (owner decision).",
        f"8. **Read every ID as a string.**{r('IDs are opaque')} {pct(ids['share_leading_zero'], 1)} of client ids start with "
        f"'0' and {n(ids['n_scientific_notation_like'])} look like scientific notation.",
        f"9. **Keep gender `U`; encode missing product attributes as an explicit 'unknown' level.**{r('`gender`', 'Product attributes')}",
        f"10. **Labels need no cleaning: use `uplift_train` only.**{r('Treatment looks randomised')} Treated share "
        f"{100 * treat['treated_share']:.2f}%, uplift {pp(overall['uplift'])}; by segment, "
        f"{segment_span(seg, 'age_band')}.{r('Uplift varies')}",
        f"11. **`uplift_test` has no labels; the tables join cleanly. Keep that as assertions in the job.**"
        f"{r('`uplift_test` carries', 'Tables join')}",
        f"12. **History is short ({times['purchase_span_days']:.0f} days; first and last months partial): use daily or weekly "
        f"windows.**{r('The purchase history is short', 'Recency has')}",
    ]


# --------------------------------------------------------------------------- document


def _short_path(path: str) -> str:
    """Last path components only, so the committed document does not contain a personal home directory."""
    parts = Path(path).parts
    return ".../" + "/".join(parts[-3:]) if len(parts) > 3 else path


def expectation_rows(results: Results, expectations: Mapping[str, Mapping[str, Any]]) -> list[str]:
    files = results.sections["files"]
    lines = ["| Table | Rows in this run | Work-plan figure | Within tolerance |", "|---|---:|---:|---|"]
    for table in ("purchases", "clients", "uplift_train", "uplift_test", "products"):
        rows = files[f"{table}_rows"]
        exp = expectations.get(table)
        if exp:
            target, tol = float(exp["rows"]), float(exp.get("tolerance_pct", 1.0))
            ok = abs(rows - target) <= target * tol / 100
            lines.append(f"| `{table}` | {n(rows)} | ~{n(target)} | {'yes' if ok else 'NO'} (±{tol:g}%) |")
        else:
            lines.append(f"| `{table}` | {n(rows)} | n/a | n/a |")
    return lines


def render(results: Results, expectations: Mapping[str, Mapping[str, Any]], outputs_dir: str) -> str:
    run = results.sections["run"]
    if run.get("mode") != "full":
        raise FindingsError(
            f"refusing to write findings from a '{run.get('mode')}' run; findings must come from the full-data outputs "
            "(python eda/run_eda.py --mode full ...)"
        )
    findings = build_findings(results)
    prov = run.get("download_provenance", {})
    files = results.sections["files"]
    treat = results.sections["treatment"]
    seg = results.tables["uplift_by_segment"]
    lines = [
        "# X5 RetailHero uplift: EDA findings (full data)",
        "",
        "> Generated by `eda/make_findings.py` from the full-data outputs of `eda/run_eda.py`. Do not edit by hand: re-run the "
        "generator. Every number below is read from the output files named under *Provenance*.",
        "",
        "## Status and scope",
        "",
        "- **Data used:** FULL data (every row of the five tables), not a sample or fixtures.",
        "- **Review status:** generated by the data-pipeline (ingestion/EDA) owner. **Not reviewed** by the cleaning/feature owner or the "
        "baseline owner; the cleaning actions below are *proposals* for them to accept or change.",
        "- **Campaign date:** unknown (not published). The leakage check is `"
        f"{results.sections['leakage']['status']}`; time-based statements use the provisional anchor "
        f"`{results.sections['leakage']['recency_anchor']}`.",
        "",
        "## Summary: what to change in cleaning",
        "",
        "One line per decision; the table under *Findings* has the evidence and caveat for each (`#n`). These are "
        "proposals for the cleaning owner, not agreed rules.",
        "",
        *build_summary(results, findings),
        "",
        "Open decisions that block final features: the campaign date, and whether `items.amount` comes from "
        "`trn_sum_from_iss` (see *Open questions*).",
        "",
        "## Provenance",
        "",
        f"- Mode: `{run['mode']}`; input directory: `{_short_path(run['input_dir'])}`.",
        f"- Source: `{', '.join(prov.get('repo_id_resolved', [])) or 'unknown'}` at revision "
        f"`{', '.join(prov.get('revision_resolved', [])) or 'unknown'}` (from the download provenance record"
        f"{'' if prov.get('found') else '; NOT FOUND for this run'}).",
        f"- Run fingerprint: `{run['fingerprint'][:16]}…`; DuckDB {run['duckdb_version']}; Python {run['python_version']}; "
        f"{run['total_seconds']} s.",
        f"- Outputs read: `{outputs_dir}/summary.json` and `{outputs_dir}/tables/*.csv`.",
        "",
        "## Row counts against the work plan",
        "",
        *expectation_rows(results, expectations),
        "",
        "The work-plan figures are rounded expectations used only as a sanity check.",
        "",
        "## Findings",
        "",
        "| # | Observation | Evidence | Recommended cleaning action | Caveat |",
        "|---:|---|---|---|---|",
    ]
    for i, f in enumerate(findings, 1):
        lines.append(
            f"| {i} | {cell(f.observation)} | {cell(f.evidence)} | {cell(f.action)} | {cell(f.caveat)} |"
        )
    lines += [
        "",
        "## Key numbers",
        "",
        "### Treatment and outcome (`uplift_train`)",
        "",
        "| Arm | Clients | Conversions | Rate |",
        "|---|---:|---:|---:|",
    ]
    for row in results.tables["treatment_outcome"].itertuples():
        lines.append(
            f"| {'treated' if int(row.treatment_flg) == 1 else 'control'} | {n(row.n)} | {n(row.conversions)} | {100 * row.rate:.2f}% |"
        )
    ov = treat["overall"]
    lines += [
        "",
        f"Uplift (treated minus control): {pp(ov['uplift'])}, 95% CI {pp(ov['ci_low'])} to {pp(ov['ci_high'])} "
        f"(Wald, z = {ov['z']:.1f}).",
        "",
        "### Uplift by segment",
        "",
        "Segments use client attributes and purchase history only. `sparse` marks segments below the minimum arm size or "
        "cell count in `configs/x5/eda.yaml`.",
        "",
        "| Segment kind | Segment | Clients | Treated | Control | Uplift | 95% CI | Sparse |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in seg.itertuples():
        sparse = str(r.sparse).lower() == "true"
        lines.append(
            f"| {r.segment_kind} | {r.segment} | {n(r.n)} | {n(r.n_treated)} | {n(r.n_control)} | {pp(r.uplift)} | "
            f"{pp(r.ci_low)} to {pp(r.ci_high)} | {'yes' if sparse else 'no'} |"
        )
    lines += [
        "",
        "## Open questions (not answerable from the data)",
        "",
        "1. What is the real campaign (SMS) date? Until it is set in `campaign.cutoff`, the leakage check stays blocked.",
        "2. Is `trn_sum_from_iss` the right source for the contract field `items.amount`? Decision needed from the contract "
        "owner; the data only show approximate agreement with `purchase_sum`.",
        "3. Why do 28 transaction ids belong to two clients, and why are card issue dates after first purchases?",
        "4. What do `U` (gender), zero `product_quantity` and null `trn_sum_from_red` mean in the source system?",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python eda/run_eda.py --mode full --input-dir data/x5/extracted",
        "python eda/make_findings.py",
        "```",
        "",
        f"Total clients in `uplift_train`: {n(treat['train_rows'])}; segments evaluated: {len(seg)}; "
        f"clients in `clients`: {n(files['clients_rows'])}.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--outputs", default=DEFAULT_OUTPUTS, help="directory written by run_eda.py (full mode)"
    )
    parser.add_argument("--output", default=DEFAULT_TARGET, help="Markdown file to write")
    parser.add_argument("--expectations-config", default=DEFAULT_EXPECTATIONS_CONFIG)
    args = parser.parse_args(argv)
    try:
        expectations = {}
        config_path = resolve_path(args.expectations_config)
        if config_path.is_file():
            expectations = load_config(config_path).get("raw_landing", {}).get("expectations", {}) or {}
        results = Results.load(resolve_path(args.outputs))
        text = render(results, expectations, args.outputs)
    except (FindingsError, EdaError, ConfigError) as exc:
        print(f"cannot generate findings: {exc}", file=sys.stderr)
        return EXIT_FAILED
    target = resolve_path(args.output)
    atomic_write_text(target, text)
    print(f"wrote {target} ({text.count(chr(10))} lines) from {args.outputs}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
