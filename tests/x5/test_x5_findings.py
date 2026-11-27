"""Tests for eda/make_findings.py (the generator of eda/FINDINGS.md).

The generator is run on results computed from the tiny hand-made dataset of ``test_x5_eda.py``. That dataset is
labelled "full" only inside these tests to exercise the rendering; its numbers are not X5 results.
"""

from __future__ import annotations

import copy
import re

import pytest
from test_x5_eda import eda_inputs, make_settings  # noqa: F401  (eda_inputs is a fixture)

from eda import make_findings as mf
from eda import x5_eda_lib as lib

HEADER = "| # | Observation | Evidence | Recommended cleaning action | Caveat |"


@pytest.fixture(scope="module")
def results(eda_inputs, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("findings_out")
    return lib.run_eda(make_settings(eda_inputs, out), figures=False, log=lambda *_: None), out


def as_full(res: lib.Results) -> lib.Results:
    clone = lib.Results()
    clone.sections = copy.deepcopy(res.sections)
    clone.tables = {k: v.copy() for k, v in res.tables.items()}
    clone.sections["run"]["mode"] = "full"
    return clone


def table_rows(text: str) -> list[list[str]]:
    start = text.index(HEADER)
    rows = []
    for line in text[start:].splitlines()[2:]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))])
    return rows


def test_sample_results_are_refused(results):
    res, _ = results
    assert res.sections["run"]["mode"] == "sample"
    with pytest.raises(mf.FindingsError, match="refusing to write findings from a 'sample' run"):
        mf.render(res, {}, "x")


def test_every_finding_has_all_four_fields(results):
    findings = mf.build_findings(as_full(results[0]))
    assert len(findings) >= 15
    for f in findings:
        assert f.observation.strip() and f.evidence.strip() and f.action.strip() and f.caveat.strip()


def test_document_structure_and_status(results):
    text = mf.render(as_full(results[0]), {}, "eda/outputs/full")
    assert HEADER in text
    assert "FULL data" in text and "**Not reviewed**" in text
    assert (
        "Prathamesh" not in text and "Yashashree" not in text and "Tejas" not in text
    )  # no claims about teammates
    assert "`blocked`" in text  # no cutoff
    assert "## Open questions" in text and "## Reproduce" in text
    rows = table_rows(text)
    assert len(rows) == len(mf.build_findings(as_full(results[0])))
    assert all(len(r) == 5 and all(r) for r in rows)
    assert [r[0] for r in rows] == [str(i) for i in range(1, len(rows) + 1)]


def test_numbers_in_the_document_come_from_the_results(results):
    res = as_full(results[0])
    text = mf.render(res, {}, "eda/outputs/full")
    s = res.sections
    assert f"{s['semantics']['row_sum_overstatement_factor']:.2f}x" in text  # 1715 / 815 = 2.10x
    assert "2.10x" in text
    assert f"{s['files']['purchases']['distinct_client_transaction_pairs']:,}" in text
    overall = s["treatment"]["overall"]
    assert f"{overall['n_treated']:,} treated / {overall['n_control']:,} control" in text
    assert f"{100 * overall['rate_treated']:.2f}% treated" in text
    # a change in the results changes the document
    res.sections["semantics"]["row_sum_overstatement_factor"] = 99.0
    assert "99.00x" in mf.render(res, {}, "eda/outputs/full")


def test_inconsistent_repeat_per_row_is_reported_differently(results):
    res = as_full(results[0])
    res.sections["semantics"]["multi_row_transactions_where_value_differs_between_rows"]["purchase_sum"] = 3
    findings = mf.build_findings(res)
    assert any("differ for some transactions" in f.observation for f in findings)
    assert not any(
        f.observation.startswith("`purchase_sum` and the four points columns repeat") for f in findings
    )


def test_expectation_table_flags_misses(results):
    res = as_full(results[0])
    lines = mf.expectation_rows(
        res, {"purchases": {"rows": 12, "tolerance_pct": 1.0}, "clients": {"rows": 1000}}
    )
    joined = "\n".join(lines)
    assert "| `purchases` | 12 | ~12 | yes (±1%) |" in joined
    assert "| `clients` | 6 | ~1,000 | NO" in joined
    assert "| `products` | 4 | n/a | n/a |" in joined


def test_cli_refuses_sample_outputs_and_missing_outputs(results, tmp_path, capsys):
    _, out = results
    target = tmp_path / "FINDINGS.md"
    assert mf.main(["--outputs", str(out), "--output", str(target)]) == mf.EXIT_FAILED
    assert "refusing" in capsys.readouterr().err and not target.exists()
    assert mf.main(["--outputs", str(tmp_path / "nothing"), "--output", str(target)]) == mf.EXIT_FAILED
    assert "no EDA results" in capsys.readouterr().err


def test_cli_writes_findings_for_full_outputs(results, tmp_path):
    res = as_full(results[0])
    out = tmp_path / "full"
    res.write(out)
    target = tmp_path / "FINDINGS.md"
    assert mf.main(["--outputs", str(out), "--output", str(target)]) == mf.EXIT_OK
    text = target.read_text()
    assert text.startswith("# X5 RetailHero uplift: EDA findings (full data)") and HEADER in text
    # the generator is deterministic for the same outputs
    first = target.read_bytes()
    assert mf.main(["--outputs", str(out), "--output", str(target)]) == mf.EXIT_OK
    assert target.read_bytes() == first


def test_cell_escapes_pipes_and_newlines():
    assert mf.cell("a|b\nc") == "a\\|b c"
    assert (
        mf.pct(1, 4) == "25.0%"
        and mf.pct(1, 0) == "n/a"
        and mf.pp(0.0332) == "+3.32 pp"
        and mf.n(1234567.4) == "1,234,567"
    )


def summary_section(text: str) -> str:
    start = text.index("## Summary: what to change in cleaning")
    return text[start : text.index("## Provenance")]


def test_summary_is_short_and_comes_before_the_details(results):
    text = mf.render(as_full(results[0]), {}, "eda/outputs/full")
    assert (
        text.index("## Status and scope")
        < text.index("## Summary")
        < text.index("## Provenance")
        < text.index("## Findings")
    )
    first_page = text[text.index("## Status and scope") : text.index("## Provenance")]
    assert len(first_page.split()) <= 450, "status + summary must fit on one page"
    items = re.findall(r"^\d+\. \*\*", summary_section(text), flags=re.M)
    assert 10 <= len(items) <= 14


def test_summary_references_point_at_matching_findings(results):
    res = as_full(results[0])
    text = mf.render(res, {}, "eda/outputs/full")
    rows = table_rows(text)
    refs = {
        int(n)
        for group in re.findall(r"\(#([\d, #]+)\)", summary_section(text))
        for n in re.findall(r"\d+", group)
    }
    assert refs and max(refs) <= len(rows)
    # the row cited for the transaction-key decision really is the transaction-key finding
    key_row = next(i for i, r in enumerate(rows, 1) if "not unique across clients" in r[1])
    assert f"(#{key_row})" in re.search(r"5\. \*\*Key a transaction.*", summary_section(text)).group(0)


def test_summary_numbers_come_from_the_results(results):
    res = as_full(results[0])
    summary = summary_section(mf.render(res, {}, "eda/outputs/full"))
    s = res.sections
    assert f"{s['semantics']['row_sum_overstatement_factor']:.2f}x" in summary
    assert f"{s['semantics']['transaction_ids_shared_by_more_than_one_client']:,} ids are shared" in summary
    assert f"{s['leakage']['status']}" in summary and s["leakage"]["recency_anchor"] in summary
    assert f"{100 * s['treatment']['treated_share']:.2f}%" in summary


def test_summary_adapts_when_repeat_per_row_assumption_fails(results):
    res = as_full(results[0])
    res.sections["semantics"]["multi_row_transactions_where_value_differs_between_rows"] = {"purchase_sum": 3}
    summary = summary_section(mf.render(res, {}, "eda/outputs/full"))
    assert "differ between product rows of some transactions" in summary
    assert "overstates spend" not in summary
