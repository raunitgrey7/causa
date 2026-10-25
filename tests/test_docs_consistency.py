"""The README results block must be the benchmark output, and no file may contain banned dashes."""

import json

import pytest

from causa.render import render_readme_section
from tests.conftest import ROOT

RESULTS = ROOT / "benchmarks" / "results.json"
BANNED = (chr(0x2014), chr(0x2013))
DOCS = ("README.md", "REPORT.md", "docs/ARCHITECTURE.md", "docs/EVALUATION.md", "benchmarks/results.md")


def test_readme_results_block_is_exactly_the_rendered_results():
    if not RESULTS.exists():
        pytest.skip("benchmarks/results.json has not been generated")
    expected = render_readme_section(json.loads(RESULTS.read_text()))
    assert expected in (ROOT / "README.md").read_text()


@pytest.mark.parametrize("path", DOCS)
def test_docs_have_no_banned_dashes(path):
    file = ROOT / path
    if not file.exists():
        pytest.skip(f"{path} not present")
    text = file.read_text()
    assert not any(ch in text for ch in BANNED), path


def test_code_and_static_files_have_no_banned_dashes():
    files = [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("tests").rglob("*.py")]
    files += [*ROOT.joinpath("src", "causa", "api", "static").iterdir()]
    files += [
        ROOT / name for name in ("Makefile", "Dockerfile", "pyproject.toml", ".github/workflows/ci.yml")
    ]
    offenders = [str(f) for f in files if f.exists() and any(ch in f.read_text() for ch in BANNED)]
    assert offenders == []


def test_thresholds_file_is_valid_and_every_check_has_a_bound():
    checks = json.loads((ROOT / "benchmarks" / "thresholds.json").read_text())["checks"]
    assert checks and all("path" in c and ("min" in c or "max" in c) for c in checks)


def test_report_contains_every_benchmark_table_row():
    report, results_md = ROOT / "REPORT.md", ROOT / "benchmarks" / "results.md"
    if not (report.exists() and results_md.exists()):
        pytest.skip("REPORT.md or benchmarks/results.md not present")
    text = report.read_text()
    rows = [line for line in results_md.read_text().splitlines() if line.startswith("|")]
    missing = [row for row in rows if row not in text]
    assert rows and missing == []
