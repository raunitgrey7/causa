"""Command line interface end to end on generated files."""

import json

from causa.cli import main


def test_generate_and_analyze_an_experiment(tmp_path, capsys):
    path = tmp_path / "exp.csv"
    assert main(["generate", "experiment", "--n", "8000", "--lift", "0.1", "--out", str(path)]) == 0
    assert main(["analyze", str(path)]) == 0
    out = capsys.readouterr().out
    assert "decision:" in out and "guardrail support_contact" in out


def test_analyze_json_output_parses(tmp_path, capsys):
    path = tmp_path / "exp.csv"
    main(["generate", "experiment", "--n", "4000", "--out", str(path)])
    capsys.readouterr()
    assert main(["analyze", str(path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["decision"] and report["primary"]["method"] == "cuped"


def test_analyze_with_a_spec_file(tmp_path, capsys):
    path, spec = tmp_path / "exp.csv", tmp_path / "spec.json"
    main(["generate", "experiment", "--n", "4000", "--out", str(path)])
    spec.write_text(json.dumps({"name": "from-file", "covariate": None, "mde": 50}))
    assert main(["analyze", str(path), "--spec", str(spec), "--json"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{") :])["experiment"] == "from-file"


def test_generate_campaign_and_estimate_the_ate(tmp_path, capsys):
    path = tmp_path / "campaign.csv"
    assert main(["generate", "campaign", "--n", "3000", "--out", str(path)]) == 0
    assert main(["ate", str(path), "--n-boot", "30"]) == 0
    out = capsys.readouterr().out
    assert "aipw" in out and "propensity AUC" in out


def test_ate_json_output_parses(tmp_path, capsys):
    path = tmp_path / "campaign.csv"
    main(["generate", "campaign", "--n", "2000", "--out", str(path)])
    capsys.readouterr()
    assert main(["ate", str(path), "--n-boot", "20", "--json"]) == 0
    assert "aipw" in json.loads(capsys.readouterr().out)["estimates"]


def test_errors_exit_with_code_2(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "missing.csv")]) == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("variant,revenue\ncontrol,1\n")
    assert main(["analyze", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err
