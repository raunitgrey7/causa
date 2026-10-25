"""Markdown rendering of ``benchmarks/results.json`` for ``results.md`` and the README.

Rendering lives apart from the harness so the docs-consistency test can re-render the README section from
the committed results file and require an exact match: a hand-edited number cannot survive.
"""

from __future__ import annotations

from typing import Any

ATE_LABELS: dict[str, tuple[str, str]] = {
    "difference_in_means": ("difference in means", "bootstrap"),
    "outcome_regression": ("outcome regression", "bootstrap"),
    "ipw": ("IPW (Hajek)", "bootstrap"),
    "aipw": ("AIPW", "bootstrap"),
    "aipw_wald": ("AIPW", "Wald (influence function)"),
    "aipw_crossfit": ("AIPW, 2-fold cross-fit", "bootstrap"),
    "aipw_crossfit_wald": ("AIPW, 2-fold cross-fit", "Wald (influence function)"),
}
CATE_ROWS: tuple[str, ...] = (
    "s_learner",
    "t_learner",
    "x_learner",
    "dr_learner",
    "constant_ate",
    "historical_rule",
    "oracle",
)


def _header(results: dict[str, Any]) -> str:
    env = results["environment"]
    return (
        f"Seed {results['seed']}, generated {results['generated_at']}, commit `{env['git_commit']}`, "
        f"Python {env['python']}, numpy {env['numpy']}, scikit-learn {env['scikit_learn']} on {env['machine']}, "
        f"{env['workers']} worker processes. Total benchmark time {results['timings_seconds']['total']} s."
    )


def _ate_table(block: dict[str, Any]) -> list[str]:
    lines = [
        "| estimator | interval | bias | SD | RMSE | coverage of 95% interval | mean width |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, (name, interval) in ATE_LABELS.items():
        e = block["estimators"][key]
        lines.append(
            f"| {name} | {interval} | {e['bias']:.2f} ({e['bias_mcse']:.2f}) | {e['sd']:.2f} | {e['rmse']:.2f} | "
            f"{e['coverage']:.3f} ({e['coverage_mcse']:.3f}) | {e['mean_width']:.1f} |"
        )
    return lines


def _ate_section(results: dict[str, Any]) -> list[str]:
    truth = results["ate"]["truth"]
    lines = [
        f"True population ATE {truth['ate']:.2f} INR per customer (Monte Carlo over {truth['draws']:,} customers, "
        f"standard error {truth['mc_se']:.3f}). Bias and coverage are followed by their Monte Carlo standard "
        "errors in brackets.",
        "",
    ]
    for name, block in results["ate"]["scenarios"].items():
        ov = block["overlap_mean"]
        lines += [
            f"#### {name} (targeting strength {block['targeting_strength']}): {block['replications']} replications "
            f"of {block['n']} customers, {block['n_boot']} refit bootstrap resamples each",
            "",
            *_ate_table(block),
            "",
            f"Overlap diagnostics, mean over replications: propensity AUC {ov['auc']:.3f}, share of customers "
            f"outside [0.05, 0.95] {ov['share_outside_band']:.3f}, effective sample size {ov['ess_frac_treated']:.3f} "
            f"(treated) and {ov['ess_frac_control']:.3f} (control) of each arm, max |SMD| {ov['max_smd_before']:.3f} "
            f"before and {ov['max_smd_after']:.3f} after weighting.",
            "",
        ]
    return lines


def _cate_section(results: dict[str, Any]) -> list[str]:
    c = results["cate"]
    lines = [
        f"{c['seeds']} seeds; each trains on a confounded history of {c['n_train']:,} customers and scores on a "
        f"randomised holdout of {c['n_holdout']:,}. Budget: coupons for {c['budget']:.0%} of customers. SD of the "
        f"true CATE {c['sd_tau']:.1f} INR; oracle policy value {c['oracle_value']:.2f} INR per customer. Values are "
        "means over seeds with the across-seed SD in brackets.",
        "",
        "| model | PEHE (INR) | PEHE / SD(tau) | Qini / oracle | AUUC | true gain area / oracle | "
        "policy value / oracle | IPW estimate of value (true value) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in CATE_ROWS:
        r = c["learners"][name]
        pehe = f"{r['pehe']:.1f} ({r['pehe_sd']:.1f}) | {r['pehe_rel']:.3f}" if "pehe" in r else "n/a | n/a"
        lines.append(
            f"| {name} | {pehe} | {r['qini_rel']:.3f} | "
            f"{r['auuc']:.2f} | {r['gain_area_rel']:.3f} | {r['policy_value_rel']:.3f} "
            f"({r['policy_value_rel_sd']:.3f}) | {r['policy_value_ipw']:.2f} ({r['policy_value']:.2f}) |"
        )
    return lines


def _budget_table(results: dict[str, Any]) -> list[str]:
    vb = results["cate"]["value_by_budget"]
    head = " | ".join(f"{b:.0%}" for b in vb["budgets"])
    lines = [f"| model | {head} |", "|---|" + "---|" * len(vb["budgets"])]
    for name, values in vb["values"].items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    return lines


def _cuped_section(results: dict[str, Any]) -> list[str]:
    c = results["cuped"]
    lines = [
        f"{c['reps']} randomised experiments of {c['n_users']:,} users, true revenue effect {c['true_effect']:.2f} "
        f"INR. Measured variance reduction {c['variance_reduction']:.3f}; mean squared pre/post correlation "
        f"(the theoretical reduction) {c['rho2']:.3f}.",
        "",
        "| estimator | bias | SD | mean SE | coverage of 95% interval | power (empirical) | power (formula) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in ("difference_in_means", "cuped"):
        e = c[key]
        lines.append(
            f"| {key} | {e['bias']:.2f} | {e['sd']:.2f} | {e['mean_se']:.2f} | {e['coverage']:.3f} | "
            f"{e['power']:.3f} | {c['predicted_power'][key]:.3f} |"
        )
    return lines


def _sequential_section(results: dict[str, Any]) -> list[str]:
    s = results["sequential"]
    labels = {
        "naive": "z-test at every look (peeking)",
        "fixed": "z-test once at the end",
        "msprt": "mixture SPRT, always-valid",
        "msprt_cuped": "mixture SPRT on CUPED-adjusted revenue",
    }
    lines = [
        f"{s['reps']} replications per scenario, {s['n_users']:,} users analysed at {s['looks']} equally spaced "
        f"looks, alpha {s['alpha']}; mixing SD tau = {s['tau']:.1f} INR (the planned effect). A/B true effect "
        f"{s['ab']['true_effect']:.2f} INR.",
        "",
        "| strategy | A/A false positive rate | A/B power |",
        "|---|---|---|",
    ]
    for key, label in labels.items():
        aa, ab = s["aa"]["reject"][key], s["ab"]["reject"][key]
        lines.append(
            f"| {label} | {aa['rate']:.4f} ({aa['mcse']:.4f}) | {ab['rate']:.4f} ({ab['mcse']:.4f}) |"
        )
    lines += [
        "",
        f"Under A/B the CUPED mixture SPRT stops after {s['ab']['msprt_cuped_users_at_decision_mean']:,.0f} users on "
        f"average (median {s['ab']['msprt_cuped_users_at_decision_median']:,.0f}) of the {s['n_users']:,} available.",
    ]
    return lines


def _srm_section(results: dict[str, Any]) -> list[str]:
    s = results["srm"]
    fb = s["fault_bias"]
    lines = [
        f"{s['reps']:,} experiments of {s['n_users']:,} users over {s['days']} days per row, alarm threshold "
        f"p < {s['alpha']}. The fault drops that share of non-converting treatment users from the logs.",
        "",
        "| fault (share lost) | chi-square once at the end | chi-square checked daily | sequential SRM, daily | report rule (either, daily) |",
        "|---|---|---|---|---|",
    ]
    for row in s["by_drop"].values():
        keys = ("chi_square_final", "chi_square_daily", "sequential", "decision_rule")
        cells = [f"{row[k]['rate']:.4f}" for k in keys]
        lines.append(f"| {row['drop']:.3f} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"With a {fb['drop']:.0%} fault the naive revenue effect averages {fb['mean_estimated_effect']:.2f} INR "
        f"against a true {fb['true_effect']:.2f} INR ({fb['reps']} experiments).",
    ]
    return lines


def render_markdown(results: dict[str, Any]) -> str:
    """The full ``benchmarks/results.md``."""
    lines = [
        "# causa benchmark results",
        "",
        _header(results),
        "",
        "## Observational ATE on a confounded campaign",
        "",
        *_ate_section(results),
        "## CATE learners and budgeted targeting",
        "",
        *_cate_section(results),
        "",
        "Incremental revenue per customer by budget (share of customers who can receive a coupon):",
        "",
        *_budget_table(results),
        "",
        "## CUPED variance reduction and power",
        "",
        *_cuped_section(results),
        "",
        "## Peeking and sequential testing",
        "",
        *_sequential_section(results),
        "",
        "## Sample ratio mismatch",
        "",
        *_srm_section(results),
        "",
        "Timings (seconds): " + ", ".join(f"{k} {v}" for k, v in results["timings_seconds"].items()) + ".",
        "",
    ]
    return "\n".join(lines)


def render_readme_section(results: dict[str, Any]) -> str:
    """The README results block: the same tables, generated so they cannot drift from ``results.json``."""
    lines = [
        _header(results),
        "",
        "### Observational ATE: bias and interval coverage against the true ATE",
        "",
        *_ate_section(results),
        "### CATE learners: PEHE, Qini, AUUC and policy value against the oracle",
        "",
        *_cate_section(results),
        "",
        "### CUPED",
        "",
        *_cuped_section(results),
        "",
        "### Peeking against the sequential test",
        "",
        *_sequential_section(results),
        "",
        "### Sample ratio mismatch detection",
        "",
        *_srm_section(results),
        "",
        "Full tables including the value-by-budget curve: [benchmarks/results.md](benchmarks/results.md).",
    ]
    return "\n".join(lines)
