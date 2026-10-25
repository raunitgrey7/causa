"""Command line interface: generate data, analyse an experiment or a campaign, train, serve, benchmark.

``causa analyze`` gives the same decision report the API returns, so an analyst can run it on a CSV
export without starting a server; ``causa ate`` does the same for an observational campaign file in the
generator's schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import uvicorn

from causa.api.app import create_app
from causa.api.schemas import ExperimentSpecIn, default_guardrails
from causa.ate import ATEConfig, estimate_ate
from causa.bench import main as bench_main
from causa.config import get_settings
from causa.data.campaign import CampaignConfig, simulate_campaign
from causa.data.experiments import ExperimentConfig, simulate_experiment
from causa.experiment.report import DecisionReport, analyze_experiment
from causa.features import FEATURE_NAMES, feature_matrix
from causa.targeting import save_model, train_targeting_model


def _generate(args: argparse.Namespace) -> int:
    if args.kind == "experiment":
        frame = simulate_experiment(
            ExperimentConfig(n_users=args.n, lift=args.lift, srm_drop=args.srm_drop, seed=args.seed)
        )
    else:
        data = simulate_campaign(CampaignConfig(n=args.n, seed=args.seed, randomized=args.randomized))
        frame = data.customers.assign(
            treatment=data.treatment, revenue=data.revenue, true_cate=data.cate.round(4)
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"wrote {len(frame)} rows to {args.out}")
    return 0


def _print_report(report: DecisionReport) -> None:
    primary, power = report.primary, report.power
    lo, hi = primary.always_valid_ci
    print(f"decision: {report.decision}")
    for reason in report.reasons:
        print(f"  {reason}")
    print(f"{report.metric} effect ({primary.method}): {primary.effect:.4g}")
    print(f"always-valid 95% interval [{lo:.4g}, {hi:.4g}], p = {primary.always_valid_p_value:.3g}")
    print(f"variance reduction {power.variance_reduction:.3f}, power at MDE {power.power_at_mde:.3f}")
    print(f"SRM p-value {report.srm.p_value:.3g} (sequential {report.srm_sequential_p_value:.3g})")
    for g in report.guardrails:
        print(f"guardrail {g.metric}: {g.status} (harm {g.harm:.4g}, margin {g.max_harm})")


def _analyze(args: argparse.Namespace) -> int:
    if args.spec:
        spec_in = ExperimentSpecIn.model_validate_json(Path(args.spec).read_text())
    else:
        spec_in = ExperimentSpecIn(name=args.csv.stem, guardrails=default_guardrails())
    report = analyze_experiment(pd.read_csv(args.csv), spec_in.to_spec())
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(report)
    return 0


def _ate(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.csv)
    result = estimate_ate(
        feature_matrix(frame),
        frame[args.treatment].to_numpy(),
        frame[args.outcome].to_numpy(float),
        ATEConfig(n_boot=args.n_boot, seed=args.seed),
        feature_names=FEATURE_NAMES,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    for name, est in result.estimates.items():
        interval = f"[{est.ci_low:9.2f}, {est.ci_high:9.2f}]"
        print(f"{name:20s} {est.estimate:9.2f}  95% CI {interval}  ({est.interval})")
    if result.overlap:
        ov = result.overlap
        print(f"propensity AUC {ov.auc:.3f}; share outside overlap band {ov.share_outside_band:.3f}")
        for warning in ov.warnings:
            print(f"warning: {warning}")
    return 0


def _train(args: argparse.Namespace) -> int:
    settings = get_settings()
    model = train_targeting_model(n_train=args.rows or settings.train_rows, seed=settings.seed)
    save_model(model, settings.model_path)
    print(f"trained {model.learner_name} on {model.n_train} customers; saved to {settings.model_path}")
    for name, scores in model.validation.items():
        print(f"  holdout Qini {name}: {scores['qini']:.3f}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    uvicorn.run(
        create_app(train_if_missing=args.train_if_missing), host=args.host, port=args.port, log_config=None
    )
    return 0


def _bench(args: argparse.Namespace) -> int:
    return bench_main(args.rest)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for every subcommand."""
    parser = argparse.ArgumentParser(prog="causa", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="write a synthetic experiment or campaign CSV")
    gen.add_argument("kind", choices=("experiment", "campaign"))
    gen.add_argument("--n", type=int, default=40_000)
    gen.add_argument("--lift", type=float, default=0.05)
    gen.add_argument("--srm-drop", type=float, default=0.0)
    gen.add_argument("--randomized", action="store_true", help="campaign only: 50/50 assignment")
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--out", type=Path, required=True)
    gen.set_defaults(func=_generate)
    ana = sub.add_parser("analyze", help="decision report for an experiment CSV")
    ana.add_argument("csv", type=Path)
    ana.add_argument("--spec", type=Path, help="JSON experiment spec (defaults match the generator)")
    ana.add_argument("--json", action="store_true")
    ana.set_defaults(func=_analyze)
    ate = sub.add_parser("ate", help="observational ATE for a campaign CSV in the generator's schema")
    ate.add_argument("csv", type=Path)
    ate.add_argument("--treatment", default="treatment")
    ate.add_argument("--outcome", default="revenue")
    ate.add_argument("--n-boot", type=int, default=200)
    ate.add_argument("--seed", type=int, default=0)
    ate.add_argument("--json", action="store_true")
    ate.set_defaults(func=_ate)
    train = sub.add_parser("train", help="train and save the targeting model")
    train.add_argument("--rows", type=int, default=None)
    train.set_defaults(func=_train)
    serve = sub.add_parser("serve", help="run the API and dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8773)
    serve.add_argument("--train-if-missing", action="store_true")
    serve.set_defaults(func=_serve)
    bench = sub.add_parser("bench", help="run the benchmark (arguments pass through)")
    bench.add_argument("rest", nargs=argparse.REMAINDER)
    bench.set_defaults(func=_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``causa`` script."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
