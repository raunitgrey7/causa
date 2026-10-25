"""Benchmark harness: determinism across worker counts, the threshold gate and the renderers."""

import json

import pytest

from causa.bench import check_thresholds, lookup, run_benchmark, update_readme, write_results
from causa.render import render_markdown, render_readme_section

VOLATILE = ("generated_at", "timings_seconds", "environment")


def _stable(results: dict) -> dict:
    out = {k: v for k, v in results.items() if k not in VOLATILE}
    for row in out["cate"]["learners"].values():
        for key in [k for k in row if k.startswith("fit_seconds")]:
            del row[key]
    return out


@pytest.fixture(scope="module")
def quick_results() -> dict:
    return run_benchmark(seed=7, workers=2, quick=True)


def test_same_seed_same_numbers_whatever_the_worker_count(quick_results):
    again = run_benchmark(seed=7, workers=1, quick=True)
    assert json.dumps(_stable(again), sort_keys=True) == json.dumps(_stable(quick_results), sort_keys=True)


def test_results_cover_every_study_named_in_the_spec(quick_results):
    assert {"ate", "cate", "cuped", "sequential", "srm"} <= set(quick_results)
    estimators = quick_results["ate"]["scenarios"]["moderate_overlap"]["estimators"]
    assert {"bias", "coverage", "coverage_mcse", "rmse"} <= set(estimators["aipw"])
    assert {"pehe", "qini_rel", "auuc", "policy_value_rel"} <= set(
        quick_results["cate"]["learners"]["dr_learner"]
    )


def test_lookup_follows_dotted_paths(quick_results):
    assert lookup(quick_results, "cuped.reps") == 200.0
    with pytest.raises(KeyError):
        lookup(quick_results, "cuped.nothing")


def test_threshold_gate_passes_and_fails_as_specified(capsys):
    results = {"a": {"b": 0.5}}
    assert check_thresholds(results, {"checks": [{"path": "a.b", "min": 0.4, "max": 0.6}]})
    assert not check_thresholds(results, {"checks": [{"path": "a.b", "max": 0.4}]})
    assert not check_thresholds(results, {"checks": [{"path": "a.missing", "min": 0.0}]})
    assert "FAIL a.missing missing" in capsys.readouterr().out


def test_markdown_and_readme_renderings(quick_results, tmp_path):
    write_results(quick_results, tmp_path)
    md = (tmp_path / "results.md").read_text()
    assert "| AIPW | bootstrap |" in md and "historical_rule | n/a" in md
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- RESULTS:BEGIN -->\nSTALE_BLOCK\n<!-- RESULTS:END -->\noutro\n")
    update_readme(quick_results, readme)
    text = readme.read_text()
    assert render_readme_section(quick_results) in text and "STALE_BLOCK" not in text
    assert text.startswith("intro") and text.rstrip().endswith("outro")


def test_rendered_markdown_has_no_banned_dashes(quick_results):
    text = render_markdown(quick_results) + render_readme_section(quick_results)
    assert chr(0x2014) not in text and chr(0x2013) not in text
