"""Tests for eda/x5_eda_lib.py, eda/run_eda.py and the notebook.

The fixture is a tiny hand-written dataset (not random, not real X5 data) whose every statistic is computed by hand
below, so these tests check the analysis code, not the data. Nothing here says anything about the real X5 data.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import nbformat
import pandas as pd
import pytest

from eda import run_eda as run_eda_cli
from eda import x5_eda_lib as lib
from eda.execute_notebook import NOTEBOOK, cell_errors, run_notebook
from eda.execute_notebook import main as execute_notebook
from ingestion.x5_common import DEFAULT_CONFIG, ConfigError, deep_merge, load_config

EDA_CONFIG = DEFAULT_CONFIG.parent / "eda.yaml"

PURCHASE_COLUMNS = [
    "client_id",
    "transaction_id",
    "transaction_datetime",
    "regular_points_received",
    "express_points_received",
    "regular_points_spent",
    "express_points_spent",
    "purchase_sum",
    "store_id",
    "product_id",
    "product_quantity",
    "trn_sum_from_iss",
    "trn_sum_from_red",
]


def _write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def purchase(client, tx, ts, ps, store, product, iss, *, rec="0.0", spent="0.0", red="", qty="1.0"):
    return [client, tx, ts, rec, "0.0", spent, "0.0", ps, store, product, qty, iss, red]


@pytest.fixture(scope="module")
def eda_inputs(tmp_path_factory) -> Path:
    """Hand-made X5-shaped tables. See the module docstring for the derived statistics."""
    root = tmp_path_factory.mktemp("eda_inputs")
    c = {i: f"000000000{i}" for i in range(1, 6)}
    c6 = "00000000a6"
    _write(
        root / "clients.csv",
        ["client_id", "first_issue_date", "first_redeem_date", "age", "gender"],
        [
            [c[1], "2018-01-01 10:00:00", "2018-02-01 10:00:00", "30", "F"],
            [c[2], "2018-03-01 10:00:00", "", "40", "M"],
            [c[3], "2018-05-01 10:00:00", "2018-04-01 00:00:00", "-5", "U"],  # redeemed before issued, invalid age
            [c[4], "2018-06-01 10:00:00", "", "150", "F"],  # invalid age
            [c[5], "2018-07-01 10:00:00", "", "28", "M"],
            [c6, "2018-08-01 10:00:00", "", "60", "F"],  # test client without purchases
        ],
    )
    _write(
        root / "uplift_train.csv",
        ["client_id", "treatment_flg", "target"],
        [[c[1], "1", "1"], [c[2], "0", "0"], [c[3], "1", "0"], [c[4], "0", "1"], [c[5], "1", "1"]],
    )
    _write(root / "uplift_test.csv", ["client_id"], [[c6]])
    _write(
        root / "products.csv",
        [
            "product_id",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "segment_id",
            "brand_id",
            "vendor_id",
            "netto",
            "is_own_trademark",
            "is_alcohol",
        ],
        [[p, "aa", "bb", "cc", "dd", "12.0", "ee", "ff", "0.5", "0", "0"] for p in ("p1", "p2", "p3", "p4")],
    )
    purchases = [
        # t1: 3 product rows, purchase_sum repeated on every row
        purchase(c[1], "t1", "2019-02-01 12:00:00", "300.0", "s1", "p1", "100.0", rec="3.0"),
        purchase(c[1], "t1", "2019-02-01 12:00:00", "300.0", "s1", "p2", "100.0", rec="3.0"),
        purchase(c[1], "t1", "2019-02-01 12:00:00", "300.0", "s1", "p3", "100.0", rec="3.0"),
        purchase(c[1], "t2", "2019-02-10 12:00:00", "50.0", "s1", "p1", "50.0"),
        # t3: iss 90 + 100 and red 10 = 200
        purchase(c[2], "t3", "2019-03-01 08:00:00", "200.0", "s2", "p2", "90.0", spent="-20.0"),
        purchase(c[2], "t3", "2019-03-01 08:00:00", "200.0", "s2", "p3", "100.0", spent="-20.0", red="10.0"),
        # t4: an exact duplicate row
        purchase(c[3], "t4", "2019-01-15 09:00:00", "100.0", "s2", "p1", "100.0"),
        purchase(c[3], "t4", "2019-01-15 09:00:00", "100.0", "s2", "p1", "100.0"),
        # the same transaction_id used by two different clients
        purchase(c[4], "t_shared", "2019-03-10 10:00:00", "70.0", "s3", "p1", "70.0"),
        purchase(c[5], "t_shared", "2019-03-12 10:00:00", "80.0", "s3", "p2", "80.0"),
        # unparseable timestamp and a product that is not in products.csv
        purchase(c[5], "t6", "not-a-date", "10.0", "s3", "p9", "10.0"),
        # a client that is not in clients.csv
        purchase("0000000099", "t7", "2019-03-11 08:00:00", "5.0", "s3", "p1", "5.0"),
    ]
    _write(root / "purchases.csv", PURCHASE_COLUMNS, purchases)
    return root


def make_settings(inputs: Path, out: Path, *, mode="sample", cutoff=None, overrides=None) -> lib.EdaSettings:
    config = load_config(EDA_CONFIG)
    config = deep_merge(
        config, {"duckdb": {"memory_limit": "1GB", "threads": 2}, "eda": {"min_arm_size": 2, "min_cell_events": 1}}
    )
    if overrides:
        config = deep_merge(config, overrides)
    return lib.EdaSettings.from_config(config, mode=mode, input_dir=inputs, output_dir=out, cutoff=cutoff)


@pytest.fixture(scope="module")
def run(eda_inputs, tmp_path_factory) -> tuple[lib.Results, Path]:
    out = tmp_path_factory.mktemp("eda_out")
    return lib.run_eda(make_settings(eda_inputs, out), log=lambda *_: None), out


# --------------------------------------------------------------------------- pure functions


def test_uplift_stats_matches_hand_calculation():
    stats = lib.uplift_stats(100, 60, 100, 50, alpha=0.05, min_arm_size=50, min_cell_events=10)
    assert stats["rate_treated"] == 0.6 and stats["rate_control"] == 0.5
    assert stats["uplift"] == pytest.approx(0.1)
    se = math.sqrt(0.6 * 0.4 / 100 + 0.5 * 0.5 / 100)
    assert stats["se"] == pytest.approx(se)
    assert stats["ci_low"] == pytest.approx(0.1 - 1.959964 * se, abs=1e-5)
    assert stats["ci_high"] == pytest.approx(0.1 + 1.959964 * se, abs=1e-5)
    assert stats["z"] == pytest.approx(0.1 / se)
    assert stats["p_value"] == pytest.approx(math.erfc(abs(0.1 / se) / math.sqrt(2)))
    assert stats["sparse"] is False and stats["n"] == 200


def test_uplift_stats_flags_sparse_and_handles_missing_arm():
    small = lib.uplift_stats(10, 5, 100, 50, alpha=0.05, min_arm_size=50, min_cell_events=1)
    assert small["sparse"] is True  # treated arm below min_arm_size
    few_events = lib.uplift_stats(100, 1, 100, 50, alpha=0.05, min_arm_size=50, min_cell_events=10)
    assert few_events["sparse"] is True  # a cell has fewer than min_cell_events
    empty = lib.uplift_stats(5, 3, 0, 0, alpha=0.05, min_arm_size=1, min_cell_events=1)
    assert math.isnan(empty["uplift"]) and math.isnan(empty["se"]) and empty["sparse"] is True
    degenerate = lib.uplift_stats(50, 50, 50, 50, alpha=0.05, min_arm_size=1, min_cell_events=0)
    assert degenerate["uplift"] == 0.0 and math.isnan(degenerate["z"])


def test_smd_and_band_labels():
    assert lib.smd(1.0, 4.0, 0.0, 4.0) == pytest.approx(0.5)
    assert math.isnan(lib.smd(1.0, 0.0, 0.0, 0.0))
    assert lib.band_labels([16, 25, 35], open_ended=False) == ["016-024", "025-034"]
    assert lib.band_labels([0, 30, 90], open_ended=True) == ["000-029", "030-089", ">=090"]


def test_parse_cutoff():
    assert lib.parse_cutoff("2019-03-01") == "2019-03-01 00:00:00"
    assert lib.parse_cutoff("2019-03-01T12:30:00") == "2019-03-01 12:30:00"
    with pytest.raises(ConfigError):
        lib.parse_cutoff("1 March")


def test_settings_precedence_and_validation(eda_inputs, tmp_path):
    config = load_config(EDA_CONFIG)
    env_dir = tmp_path / "env"
    s = lib.EdaSettings.from_config(config, mode="full", env={"X5_EDA_INPUT_DIR": str(env_dir)})
    assert s.input_dir == env_dir and s.mode == "full"
    s = lib.EdaSettings.from_config(config, mode="full", input_dir=eda_inputs, env={"X5_EDA_INPUT_DIR": str(env_dir)})
    assert s.input_dir == eda_inputs  # CLI/argument beats env
    s = lib.EdaSettings.from_config(config, mode="sample")
    assert s.input_dir.name == "raw_sample" and s.output_dir.parts[-3:] == ("eda", "outputs", "sample")  # YAML default
    assert s.cutoff is None and s.alpha == 0.05
    assert lib.EdaSettings.from_config(config, mode="sample", cutoff="2019-03-01").cutoff == "2019-03-01"
    with pytest.raises(ConfigError):
        lib.EdaSettings.from_config(config, mode="bogus")
    with pytest.raises(ConfigError):
        lib.EdaSettings.from_config(
            deep_merge(config, {"eda": {"segments": {"age_band_edges": [30, 20]}}}), mode="sample"
        )
    with pytest.raises(ConfigError):
        lib.EdaSettings.from_config(deep_merge(config, {"eda": {"alpha": 2}}), mode="sample")
    with pytest.raises(ConfigError):
        lib.EdaSettings.from_config(config, mode="sample", cutoff="not a date")
    with pytest.raises(ConfigError):
        lib.EdaSettings.from_config({"x": 1}, mode="sample")


def test_results_roundtrip_keeps_text_ids_and_turns_nan_into_null(tmp_path):
    res = lib.Results()
    res.section("a")["value"] = float("nan")
    res.section("a")["nested"] = {"inf": float("inf"), "n": 3}
    res.add_table(
        "t", pd.DataFrame({"client_id": ["0001", "0002"], "x": [1.5, float("nan")], "segment": ["010", "020"]})
    )
    res.write(tmp_path)
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["sections"]["a"] == {"value": None, "nested": {"inf": None, "n": 3}}
    loaded = lib.Results.load(tmp_path)
    assert loaded.tables["t"]["client_id"].tolist() == ["0001", "0002"]  # leading zeros survive
    assert loaded.tables["t"]["segment"].tolist() == ["010", "020"]
    assert loaded.tables["t"]["x"].iloc[0] == 1.5 and math.isnan(loaded.tables["t"]["x"].iloc[1])
    with pytest.raises(lib.EdaError):
        lib.Results.load(tmp_path / "missing")


def test_flatten_nested_scalars():
    frame = lib.flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3})
    assert dict(zip(frame["key"], frame["value"], strict=True)) == {"a.b": 1, "a.c.d": 2, "e": 3}


# --------------------------------------------------------------------------- run on the hand-made data


def test_mode_is_labelled_and_inputs_recorded(run):
    res, _ = run
    info = res.sections["run"]
    assert info["mode"] == "sample" and "SAMPLE" in info["mode_description"]
    assert set(info["input_files"]) == set(lib.REQUIRED_COLUMNS)
    assert info["duckdb_version"] and info["fingerprint"]


def test_file_counts_keys_and_relationships(run):
    f = run[0].sections["files"]
    assert (
        f["clients_rows"],
        f["uplift_train_rows"],
        f["uplift_test_rows"],
        f["products_rows"],
        f["purchases_rows"],
    ) == (6, 5, 1, 4, 12)
    assert all(v["duplicate_rows"] == 0 for v in f["key_uniqueness"].values())
    p = f["purchases"]
    assert p["distinct_transaction_id"] == 7 and p["distinct_client_transaction_pairs"] == 8
    assert p["exact_duplicate_rows_hash_based"] == 1
    assert p["rows_beyond_first_per_transaction_product_pair"] == 1
    r = f["relationships"]
    assert r["purchase_clients_not_in_clients"] == 1 and r["purchase_rows_with_client_not_in_clients"] == 1
    assert r["purchase_product_ids_not_in_products"] == 1 and r["purchase_rows_with_product_not_in_products"] == 1
    assert r["products_never_purchased"] == 1  # p4
    assert r["train_test_overlap_clients"] == 0 and r["train_clients_not_in_clients"] == 0
    assert r["clients_in_neither_train_nor_test"] == 0
    assert f["cast_failures_total"] == 1  # the "not-a-date" timestamp, nothing else


def test_cast_failures_name_the_column(run):
    cast = run[0].tables["cast_failures"]
    bad = cast[cast["n_failed"] > 0]
    assert bad[["table", "column"]].values.tolist() == [["purchases", "transaction_datetime"]]


def test_timestamps(run):
    t = run[0].sections["timestamps"]
    assert str(t["purchase_ts_min"]) == "2019-01-15 09:00:00" and str(t["purchase_ts_max"]) == "2019-03-12 10:00:00"
    assert t["purchase_ts_null_or_invalid_rows"] == 1 and t["purchase_ts_unparseable_non_null_values"] == 1
    assert t["clients_first_redeem_before_first_issue"] == 1
    assert t["clients_first_redeem_null"] == 4
    assert t["clients_first_issue_after_first_purchase"] == 0
    months = run[0].tables["transactions_by_month"]
    assert months.set_index("month")["n_transactions"].to_dict() == {
        "2019-01": 1,
        "2019-02": 2,
        "2019-03": 4,
        "<no_timestamp>": 1,
    }


def test_leakage_is_blocked_without_cutoff(run):
    leak = run[0].sections["leakage"]
    assert leak["status"] == "blocked" and "campaign.cutoff is not set" in leak["reason"]
    assert leak["recency_anchor"] == "2019-03-12 10:00:00" and "PROVISIONAL" in leak["recency_anchor_label"]


def test_leakage_with_cutoff_counts_rows_and_anchors_recency(eda_inputs, tmp_path):
    res = lib.run_eda(make_settings(eda_inputs, tmp_path, cutoff="2019-02-15"), figures=False, log=lambda *_: None)
    leak = res.sections["leakage"]
    assert leak["status"] == "checked" and leak["cutoff"] == "2019-02-15 00:00:00"
    assert leak["purchase_rows_at_or_after_cutoff"] == 5  # t3 x2, t_shared x2, t7 (null timestamps never count)
    assert leak["transactions_at_or_after_cutoff"] == 4
    assert leak["recency_anchor"] == "2019-02-15 00:00:00" and "cutoff" in leak["recency_anchor_label"]
    assert leak["clients_first_redeem_at_or_after_cutoff"] == 0


def test_treatment_and_overall_uplift(run):
    t = run[0].sections["treatment"]
    o = t["overall"]
    assert (o["n_treated"], o["conversions_treated"], o["n_control"], o["conversions_control"]) == (3, 2, 2, 1)
    assert o["uplift"] == pytest.approx(2 / 3 - 1 / 2)
    assert t["treated_share"] == pytest.approx(0.6)
    assert t["rows_with_invalid_treatment_flg"] == 0 and t["rows_with_invalid_target"] == 0
    balance = run[0].tables["covariate_balance"].set_index("covariate")
    assert balance.loc["age_valid", "n_treated"] == 2  # c3's age -5 is invalid
    assert balance.loc["age_valid", "mean_treated"] == pytest.approx(29.0)
    assert balance.loc["gender_female", "mean_treated"] == pytest.approx(1 / 3)


def test_segments_use_hand_computed_groups(run):
    res = run[0]
    seg = res.tables["uplift_by_segment"]

    def row(kind, name):
        return seg[(seg["segment_kind"] == kind) & (seg["segment"] == name)].iloc[0]

    young = row("age_band", "025-034")  # c1 (T,1) and c5 (T,1)
    assert (young["n_treated"], young["n_control"], young["conversions_treated"]) == (2, 0, 2)
    assert math.isnan(young["uplift"]) and bool(young["sparse"])
    invalid = row("age_band", lib.INVALID_AGE)  # c3 age -5 (T,0) and c4 age 150 (C,1)
    assert (invalid["n_treated"], invalid["n_control"], invalid["uplift"]) == (1, 1, -1.0)
    assert row("age_band", "035-044")["n"] == 1
    edges = res.sections["segments"]["spend_tier_edges"]
    assert edges == {
        "spend_Q1": pytest.approx(90.0),
        "spend_Q2": pytest.approx(100.0),
        "spend_Q3": pytest.approx(200.0),
    }
    assert row("spend_tier", "spend_Q1")["n"] == 2  # c4 (70) and c5 (90)
    assert row("spend_tier", "spend_Q4")["n"] == 1  # c1 (350)
    assert row("recency_band", "000-014")["n"] == 3  # c2 (11.1 days), c4 (2.0), c5 (0)
    assert row("recency_band", "015-029")["n"] == 1  # c1 (29.9 days)
    assert row("recency_band", "030-059")["n"] == 1  # c3 (56.0 days)
    assert row("gender", "F")["n"] == 2
    assert seg.groupby("segment_kind")["n"].sum().to_dict() == {
        "age_band": 5,
        "gender": 5,
        "recency_band": 5,
        "spend_tier": 5,
    }


def test_distributions_and_points_signs(run):
    res = run[0]
    d = res.sections["distributions"]
    spend = d["spend_per_transaction"]
    assert spend["n_non_null"] == 8 and spend["min"] == 5.0 and spend["max"] == 300.0
    assert spend["mean"] == pytest.approx(815 / 8)  # purchase_sum counted once per (client, transaction)
    assert d["transactions_per_client_with_purchases"]["max"] == 2
    assert d["age_outside_valid_range"] == {"below_min": 1, "above_max": 1, "null": 0}
    points = res.tables["points_sign_conventions"].set_index("column")
    assert points.loc["regular_points_spent", "n_negative"] == 1 and points.loc["regular_points_spent", "min"] == -20.0
    assert (
        points.loc["regular_points_received", "n_positive"] == 1 and points.loc["regular_points_received", "max"] == 3.0
    )
    assert res.tables["gender_counts"].set_index("gender")["n_clients"].to_dict() == {"F": 3, "M": 2, "U": 1}


def test_semantics_detects_repeated_purchase_sum_and_shared_ids(run):
    s = run[0].sections["semantics"]
    assert s["transactions"] == 8 and s["product_rows"] == 12
    assert s["transactions_with_several_rows"] == 3 and s["transactions_with_one_row"] == 5
    assert s["sum_purchase_sum_over_rows"] == pytest.approx(1715.0)
    assert s["sum_purchase_sum_once_per_transaction"] == pytest.approx(815.0)
    assert s["row_sum_overstatement_factor"] == pytest.approx(1715 / 815)
    assert s["interpretation"]["purchase_sum_and_points_are_transaction_level"] is True
    assert all(v == 0 for v in s["multi_row_transactions_where_value_differs_between_rows"].values())
    assert s["transaction_ids_shared_by_more_than_one_client"] == 1
    assert (
        s["client_transactions_with_shared_transaction_id"] == 2 and s["product_rows_with_shared_transaction_id"] == 2
    )
    line = s["line_amount_vs_purchase_sum"]
    assert line["sum_iss_equals_purchase_sum"] == 6 and line["sum_iss_plus_red_equals_purchase_sum"] == 7
    assert line["sum_iss_above_purchase_sum"] == 1 and line["sum_iss_below_purchase_sum"] == 1  # t4 duplicate, t3
    assert s["rows_with_trn_sum_from_red"] == 1 and s["rows_with_nan_or_inf_numeric"] == 0
    assert s["transactions_with_nonpositive_purchase_sum"] == 0


def test_repeated_purchase_sum_that_differs_between_rows_is_reported(tmp_path):
    root = tmp_path / "in"
    root.mkdir()
    _write(
        root / "clients.csv",
        ["client_id", "first_issue_date", "first_redeem_date", "age", "gender"],
        [["a", "2018-01-01 00:00:00", "", "30", "F"]],
    )
    _write(root / "uplift_train.csv", ["client_id", "treatment_flg", "target"], [["a", "1", "1"]])
    _write(root / "uplift_test.csv", ["client_id"], [])
    _write(root / "products.csv", ["product_id"], [["p1"], ["p2"]])
    _write(
        root / "purchases.csv",
        PURCHASE_COLUMNS,
        [
            purchase("a", "t1", "2019-01-01 00:00:00", "10.0", "s", "p1", "5.0"),
            purchase("a", "t1", "2019-01-01 00:00:00", "12.0", "s", "p2", "5.0"),
        ],
    )
    res = lib.run_eda(make_settings(root, tmp_path / "out"), figures=False, log=lambda *_: None)
    s = res.sections["semantics"]
    assert s["multi_row_transactions_where_value_differs_between_rows"]["purchase_sum"] == 1
    assert s["interpretation"]["purchase_sum_and_points_are_transaction_level"] is False


def test_id_formats_and_uplift_test_and_unmatched(run):
    res = run[0]
    ids = res.tables["id_formats"].set_index(["table", "column"])
    row = ids.loc[("clients", "client_id")]
    assert row["distinct_values"] == 6 and row["min_length"] == row["max_length"] == 10
    assert row["n_scientific_notation_like"] == 0 and row["n_blank_or_padded"] == 0
    assert row["share_leading_zero"] == 1.0 and row["share_digits_only"] == pytest.approx(5 / 6)
    assert res.sections["id_formats"]["products_segment_id_integer_valued_floats"] == 4
    u = res.sections["uplift_test"]
    assert u["has_labels"] is False and u["columns"] == ["client_id"] and u["clients"] == 1
    assert u["clients_with_purchases"] == 0
    without = res.tables["clients_without_purchases"]
    assert int(without["n_without_purchases"].sum()) == 1
    assert without[without["population"] == "uplift_test"]["n_without_purchases"].iloc[0] == 1


def test_written_tables_keep_leading_zeros(run):
    _, out = run
    text = (out / "tables" / "id_formats.csv").read_text()
    assert "client_id" in text
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["tables"]) >= {"uplift_by_segment", "null_counts", "cast_failures", "covariate_balance"}
    assert (out / "tables" / "uplift_by_segment.csv").read_text().count("\n") > 10


def test_figures_are_written(run):
    _, out = run
    names = {p.name for p in (out / "figures").glob("*.png")}
    assert {"uplift_by_age_band.png", "covariate_balance.png", "spend_per_transaction.png"} <= names
    assert all((out / "figures" / n).stat().st_size > 1000 for n in names)


def test_typed_tables_are_reused_and_rebuilt_on_change(eda_inputs, tmp_path):
    logs: list[str] = []
    settings = make_settings(eda_inputs, tmp_path)
    lib.run_eda(settings, figures=False, log=logs.append)
    assert any("built" in m and "reused" not in m for m in logs)
    logs.clear()
    again = lib.run_eda(settings, figures=False, log=logs.append)
    assert any("reused" in m for m in logs)
    assert again.sections["files"]["purchases_rows"] == 12
    logs.clear()
    lib.run_eda(settings, force=True, figures=False, log=logs.append)
    assert any("built" in m and "reused" not in m for m in logs)
    logs.clear()
    lib.run_eda(make_settings(eda_inputs, tmp_path, cutoff="2019-02-15"), figures=False, log=logs.append)
    assert any("built" in m and "reused" not in m for m in logs)  # settings changed -> fingerprint changed


def _assert_same(left, right, path=""):
    """Equal structure and values; floats may differ in the last bits (multi-threaded sums are not bit-stable)."""
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path
        for key in left:
            _assert_same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), path
        for i, (x, y) in enumerate(zip(left, right, strict=True)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(left, float):
        assert left == pytest.approx(right, rel=1e-9, abs=1e-12), path
    else:
        assert left == right, path


def test_results_are_reproducible(eda_inputs, tmp_path):
    a = lib.run_eda(make_settings(eda_inputs, tmp_path / "a"), figures=False, log=lambda *_: None)
    b = lib.run_eda(make_settings(eda_inputs, tmp_path / "b"), figures=False, log=lambda *_: None)
    for name in a.tables:
        pd.testing.assert_frame_equal(a.tables[name], b.tables[name], check_exact=False, rtol=1e-9, atol=1e-12)
    for section in ("files", "treatment", "semantics", "segments", "distributions", "timestamps", "leakage"):
        _assert_same(lib._clean(a.sections[section]), lib._clean(b.sections[section]), section)


def test_missing_inputs_and_columns_fail_clearly(eda_inputs, tmp_path):
    with pytest.raises(lib.EdaError, match="does not exist"):
        lib.run_eda(make_settings(tmp_path / "nope", tmp_path / "o1"), figures=False, log=lambda *_: None)
    partial = tmp_path / "partial"
    partial.mkdir()
    for name in ("clients", "uplift_train", "uplift_test", "products"):
        (partial / f"{name}.csv").write_text((eda_inputs / f"{name}.csv").read_text())
    with pytest.raises(lib.EdaError, match="purchases"):
        lib.run_eda(make_settings(partial, tmp_path / "o2"), figures=False, log=lambda *_: None)
    broken = tmp_path / "broken"
    broken.mkdir()
    for name in ("clients", "uplift_train", "uplift_test", "products", "purchases"):
        (broken / f"{name}.csv").write_text((eda_inputs / f"{name}.csv").read_text())
    (broken / "uplift_train.csv").write_text("client_id,treatment_flg\n0000000001,1\n")
    with pytest.raises(lib.EdaError, match="missing required columns"):
        lib.run_eda(make_settings(broken, tmp_path / "o3"), figures=False, log=lambda *_: None)


def test_gzip_inputs_are_read(eda_inputs, tmp_path):
    import gzip
    import shutil

    gz = tmp_path / "gz"
    gz.mkdir()
    for path in eda_inputs.glob("*.csv"):
        with path.open("rb") as src, gzip.open(gz / f"{path.name}.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
    res = lib.run_eda(make_settings(gz, tmp_path / "out"), figures=False, log=lambda *_: None)
    assert res.sections["files"]["purchases_rows"] == 12


# --------------------------------------------------------------------------- CLI and notebook


def test_cli_exit_codes(eda_inputs, tmp_path, capsys):
    out = tmp_path / "out"
    ok = run_eda_cli.main(
        ["--mode", "sample", "--input-dir", str(eda_inputs), "--output-dir", str(out), "--no-figures"], env={}
    )
    assert ok == run_eda_cli.EXIT_OK and (out / "summary.json").is_file()
    assert (
        run_eda_cli.main(["--input-dir", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "o")], env={})
        == run_eda_cli.EXIT_FAILED
    )
    assert run_eda_cli.main(["--cutoff", "tomorrow", "--input-dir", str(eda_inputs)], env={}) == run_eda_cli.EXIT_USAGE
    assert "configuration error" in capsys.readouterr().err
    assert run_eda_cli.main(["--config", str(tmp_path / "no.yaml")], env={}) == run_eda_cli.EXIT_USAGE


def test_cli_env_input_dir(eda_inputs, tmp_path):
    out = tmp_path / "out"
    code = run_eda_cli.main(
        ["--mode", "full", "--output-dir", str(out), "--no-figures"], env={"X5_EDA_INPUT_DIR": str(eda_inputs)}
    )
    assert code == 0
    assert json.loads((out / "summary.json").read_text())["sections"]["run"]["mode"] == "full"


def test_notebook_is_clean_and_covers_all_sections():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, "committed notebook must not contain error outputs"
    headings = [
        line for c in nb.cells if c.cell_type == "markdown" for line in c.source.splitlines() if line.startswith("## ")
    ]
    for number in range(1, 11):
        assert any(h.startswith(f"## {number}. ") for h in headings), f"section {number} missing"
    source = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "MODE" in source and "run_eda(" in source


def test_notebook_executes_headlessly_on_the_hand_made_data(eda_inputs, tmp_path):
    out = tmp_path / "nb_out"
    executed = run_notebook("sample", input_dir=eda_inputs, output_dir=out, timeout=300)
    assert cell_errors(executed) == []
    text = "".join(
        o.get("text", "")
        for c in executed.cells
        if c.cell_type == "code"
        for o in c.outputs
        if o.output_type == "stream"
    )
    assert "MODE: SAMPLE" in text
    assert (out / "summary.json").is_file()
    assert not list(tmp_path.rglob("*.ipynb")), "no executed notebook copy may be written"


def test_execute_cli_writes_no_notebook_copy(eda_inputs, tmp_path):
    out = tmp_path / "cli_out"
    code = execute_notebook(
        ["--mode", "sample", "--input-dir", str(eda_inputs), "--output-dir", str(out), "--timeout", "300"]
    )
    assert code == 0
    assert (out / "summary.json").is_file()
    assert not list(tmp_path.rglob("*.ipynb"))


def test_provenance_summary_links_inputs_to_download_record(eda_inputs, tmp_path):
    files = {t: eda_inputs / f"{t}.csv" for t in lib.REQUIRED_COLUMNS}
    assert lib._provenance_summary(eda_inputs, files) == {"found": False}
    root = tmp_path / "data"
    (root / "downloads").mkdir(parents=True)
    artifacts = {
        t: {
            "sha256": f"sha-{t}",
            "bytes": 1,
            "extracted_members": [{"bytes": files[t].stat().st_size}],
            "source": {"revision_resolved": "rev1", "repo_id_resolved": "org/repo"},
        }
        for t in files
    }
    (root / "downloads" / "download_provenance.json").write_text(json.dumps({"artifacts": artifacts}))
    (root / "extracted").mkdir()
    summary = lib._provenance_summary(root / "extracted", files)
    assert summary["found"] and summary["revision_resolved"] == ["rev1"] and summary["repo_id_resolved"] == ["org/repo"]
    assert all(
        v["input_size_matches_extracted_member"] and not v["input_size_matches_artifact"]
        for v in summary["tables"].values()
    )
    (root / "downloads" / "download_provenance.json").write_text("not json")
    assert lib._provenance_summary(root / "extracted", files)["found"] is False
