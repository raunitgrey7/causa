# REPORT: causa

Causal inference and experimentation for growth, built as a portfolio project under my
project charter and spec. This file records what was built, every measured
number with the command that produced it, what went wrong on the way and what is still missing.

## 1. What was built

* **Data with ground truth** (`causa.data`). A seeded campaign generator: customers from three latent
  factors, a confounded historical coupon assignment, and closed-form potential outcome means, so every
  customer's CATE and the population ATE are known. A seeded A/B generator: gamma-Poisson spending with a
  pre-period window for CUPED, two guardrail metrics, and a logging fault that causes a sample ratio
  mismatch.
* **Observational estimators** (`propensity`, `ate`, `linear`). Propensity model with overlap and balance
  diagnostics (AUC, share outside [0.05, 0.95], effective sample size, SMD before and after weighting, with
  warnings); difference in means, outcome regression, Hajek IPW, AIPW and 2-fold cross-fitted AIPW, each with
  a percentile interval from a nonparametric bootstrap that refits every nuisance model, plus Wald intervals
  for both AIPW forms. The refits are batched: all resamples of a weighted ridge or logistic regression are
  solved in one matrix product.
* **CATE and targeting** (`learners`, `uplift`, `targeting`). S, T, X and DR learners on one gradient
  boosting base learner; Qini curve and coefficient, uplift curve and AUUC, true gain curve; a budgeted
  policy with true and IPW-estimated policy value; a serving model chosen by Qini on a randomised holdout.
* **Experiment analysis** (`causa.experiment`). Difference in means and CUPED from per-arm sufficient
  statistics, a delta-method relative lift, the mixture SPRT with always-valid p-values and confidence
  sequences, chi-square and sequential SRM tests, power and sample size, and a decision report with five
  documented verdicts.
* **Evaluation** (`causa.studies`, `bench`, `render`). Monte Carlo and replication studies of every method
  against the ground truth, written to `benchmarks/results.json` and `results.md`, copied into the README,
  and gated by `benchmarks/thresholds.json`.
* **Service** (`causa.api`, `cli`). FastAPI with typed schemas, `/health`, Prometheus `/metrics`, JSON
  access logs with request ids, input validation and limits, optional API key; a sqlite store of decision
  reports; a vanilla JS dashboard; a CLI. Packaging: Makefile, multi-stage non-root Dockerfile, GitHub
  Actions workflow with lint, tests, the benchmark gate and a container health check.

## 2. Spec coverage

| spec item | where | evidence |
|---|---|---|
| customers with covariates, confounded assignment, ground-truth CATE | `data/campaign.py` | `tests/test_campaign.py` (closed-form CATE, confounding visible, removed by randomisation) |
| randomised A/B datasets with pre-period metrics | `data/experiments.py` | `tests/test_experiment_data.py` (closed-form means match simulation) |
| propensity modelling and overlap checks | `propensity.py` | `tests/test_propensity.py`; overlap rows in the ATE tables |
| IPW and doubly robust ATE with bootstrap intervals | `ate.py`, `linear.py` | `tests/test_ate.py`, `tests/test_linear.py`; ATE tables (500 replications per scenario) |
| T-, S-, X- and DR-learners | `learners.py` | `tests/test_learners.py`; CATE table (5 seeds) |
| uplift-based targeting under a budget | `uplift.py`, `targeting.py` | `tests/test_uplift.py`, `tests/test_targeting.py`; policy value columns |
| CUPED | `experiment/stats.py` | `tests/test_experiment_stats.py`; CUPED table (2,000 replications) |
| sequential testing with always-valid p-values | `experiment/sequential.py` | `tests/test_sequential.py`; peeking table (2,000 replications per scenario) |
| SRM detection | `experiment/srm.py`, `experiment/sequential.py` | `tests/test_srm_power.py`; SRM table (10,000 experiments per fault size) |
| ATE bias and coverage, PEHE, Qini, AUUC, policy value vs oracle, CUPED variance reduction, peeking FPR vs sequential | `studies/`, `bench.py` | section 3 |
| upload an experiment, decision report (effect, interval, power, guardrails) | `experiment/report.py`, `POST /v1/experiments` and `/csv` | `tests/test_report.py`, `tests/test_api.py` |
| targeting endpoint | `POST /v1/targeting` | `tests/test_api.py` |
| dashboard | `api/static/` | served at `/`; exercised over HTTP in section 8 |

## 3. Measured results

Command: `make bench` (that is, `uv run python -m causa.bench --check --update-readme`), seed 42, two worker
processes. The tables below are `benchmarks/results.md` from the final run, inserted verbatim;
`tests/test_docs_consistency.py` checks that every table row of `results.md` appears here and that the
README block is exactly the rendering of `results.json`.

<!-- RESULTS-MD:BEGIN -->
Seed 42, generated 2026-10-25T05:52:19+00:00, commit `2b6b3f9`, Python 3.13.17, numpy 2.5.4, scikit-learn 1.9.1 on Darwin arm64, 2 worker processes. Total benchmark time 268.31 s.

### Observational ATE on a confounded campaign

True population ATE 20.62 INR per customer (Monte Carlo over 4,000,000 customers, standard error 0.064); the CATE has SD 128.1 INR and is negative for 46.1% of customers. Bias and coverage are followed by their Monte Carlo standard errors in brackets.

##### moderate_overlap (targeting strength 1.0): 500 replications of 4000 customers, 200 refit bootstrap resamples each

| estimator | interval | bias | SD | RMSE | coverage of 95% interval | mean width |
|---|---|---|---|---|---|---|
| difference in means | bootstrap | 165.22 (1.43) | 32.06 | 168.30 | 0.000 (0.000) | 119.1 |
| outcome regression | bootstrap | 3.81 (1.49) | 33.40 | 33.58 | 0.928 (0.012) | 127.4 |
| IPW (Hajek) | bootstrap | -2.49 (1.65) | 36.80 | 36.84 | 0.938 (0.011) | 136.8 |
| AIPW | bootstrap | -1.77 (1.59) | 35.53 | 35.54 | 0.936 (0.011) | 131.9 |
| AIPW | Wald (influence function) | -1.77 (1.59) | 35.53 | 35.54 | 0.942 (0.010) | 132.7 |
| AIPW, 2-fold cross-fit | bootstrap | -4.78 (2.08) | 46.48 | 46.68 | 0.986 (0.005) | 269.9 |
| AIPW, 2-fold cross-fit | Wald (influence function) | -4.78 (2.08) | 46.48 | 46.68 | 0.952 (0.010) | 171.5 |

Overlap diagnostics, mean over replications: propensity AUC 0.790, share of customers outside [0.05, 0.95] 0.038, effective sample size 0.518 (treated) and 0.713 (control) of each arm, max |SMD| 0.846 before and 0.043 after weighting.

##### weak_overlap (targeting strength 1.8): 500 replications of 4000 customers, 200 refit bootstrap resamples each

| estimator | interval | bias | SD | RMSE | coverage of 95% interval | mean width |
|---|---|---|---|---|---|---|
| difference in means | bootstrap | 233.42 (1.39) | 31.02 | 235.47 | 0.000 (0.000) | 125.5 |
| outcome regression | bootstrap | 9.42 (1.97) | 44.15 | 45.10 | 0.938 (0.011) | 173.6 |
| IPW (Hajek) | bootstrap | 9.99 (2.75) | 61.39 | 62.13 | 0.926 (0.012) | 204.6 |
| AIPW | bootstrap | 3.64 (2.57) | 57.56 | 57.62 | 0.934 (0.011) | 200.2 |
| AIPW | Wald (influence function) | 3.64 (2.57) | 57.56 | 57.62 | 0.914 (0.013) | 192.2 |
| AIPW, 2-fold cross-fit | bootstrap | -0.55 (3.32) | 74.24 | 74.16 | 0.990 (0.004) | 398.9 |
| AIPW, 2-fold cross-fit | Wald (influence function) | -0.55 (3.32) | 74.24 | 74.16 | 0.948 (0.010) | 259.8 |

Overlap diagnostics, mean over replications: propensity AUC 0.891, share of customers outside [0.05, 0.95] 0.265, effective sample size 0.209 (treated) and 0.392 (control) of each arm, max |SMD| 1.210 before and 0.135 after weighting.

### CATE learners and budgeted targeting

5 seeds; each trains on a confounded history of 50,000 customers and scores on a randomised holdout of 40,000. Budget: coupons for 20% of customers. SD of the true CATE 127.5 INR; oracle policy value 40.69 INR per customer. Values are means over seeds with the across-seed SD in brackets.

| model | PEHE (INR) | PEHE / SD(tau) | Qini / oracle | AUUC | true gain area / oracle | policy value / oracle | IPW estimate of value (true value) |
|---|---|---|---|---|---|---|---|
| s_learner | 83.9 (8.5) | 0.658 | 0.939 | 38.64 | 0.909 | 0.913 (0.015) | 39.32 (37.16) |
| t_learner | 140.3 (7.1) | 1.100 | 0.660 | 30.23 | 0.728 | 0.714 (0.063) | 30.71 (29.04) |
| x_learner | 87.1 (11.1) | 0.683 | 0.869 | 36.50 | 0.861 | 0.871 (0.030) | 36.75 (35.44) |
| dr_learner | 108.4 (11.0) | 0.850 | 0.750 | 32.92 | 0.820 | 0.801 (0.062) | 31.16 (32.58) |
| constant_ate | 128.1 (0.4) | 1.004 | -0.001 | 9.59 | 0.241 | 0.098 (0.003) | 3.81 (4.00) |
| historical_rule | n/a | n/a | 0.141 | 13.91 | 0.298 | 0.137 (0.006) | 8.56 (5.59) |
| oracle | 0.0 (0.0) | 0.000 | 1.000 | 40.67 | 1.000 | 1.000 (0.000) | 42.63 (40.69) |

Incremental revenue per customer by budget (share of customers who can receive a coupon):

| model | 5% | 10% | 20% | 30% | 50% | 100% |
|---|---|---|---|---|---|---|
| oracle | 18.79 | 28.74 | 40.69 | 47.08 | 51.40 | 51.48 |
| s_learner | 15.59 | 25.03 | 37.16 | 43.87 | 48.58 | 47.46 |
| t_learner | 12.57 | 20.06 | 29.04 | 33.72 | 36.69 | 36.81 |
| x_learner | 15.33 | 24.28 | 35.44 | 41.07 | 44.49 | 44.35 |
| dr_learner | 13.85 | 22.01 | 32.58 | 38.91 | 43.33 | 43.21 |

### CUPED variance reduction and power

2000 randomised experiments of 20,000 users, true revenue effect 78.40 INR. Measured variance reduction 0.342; mean squared pre/post correlation (the theoretical reduction) 0.349.

| estimator | bias | SD | mean SE | coverage of 95% interval | power (empirical) | power (formula) |
|---|---|---|---|---|---|---|
| difference_in_means | -0.33 | 30.99 | 30.14 | 0.945 | 0.724 | 0.739 |
| cuped | 0.03 | 25.13 | 24.31 | 0.947 | 0.893 | 0.897 |

### Peeking and sequential testing

2000 replications per scenario, 20,000 users analysed at 40 equally spaced looks, alpha 0.05; mixing SD tau = 78.4 INR (the planned effect). A/B true effect 78.40 INR.

| strategy | A/A false positive rate | A/B power |
|---|---|---|
| z-test at every look (peeking) | 0.2875 (0.0101) | 0.8540 (0.0079) |
| z-test once at the end | 0.0565 (0.0052) | 0.7320 (0.0099) |
| mixture SPRT, always-valid | 0.0140 (0.0026) | 0.4330 (0.0111) |
| mixture SPRT on CUPED-adjusted revenue | 0.0115 (0.0024) | 0.6610 (0.0106) |

Under A/B the CUPED mixture SPRT stops after 14,351 users on average (median 15,500) of the 20,000 available.

### Sample ratio mismatch

10,000 experiments of 100,000 users over 14 days per row, alarm threshold p < 0.001. The fault drops that share of non-converting treatment users from the logs.

| fault (share lost) | chi-square once at the end | chi-square checked daily | sequential SRM, daily | report rule (either, daily) |
|---|---|---|---|---|
| 0.000 | 0.0016 | 0.0070 | 0.0004 | 0.0070 |
| 0.005 | 0.0026 | 0.0116 | 0.0005 | 0.0116 |
| 0.010 | 0.0092 | 0.0240 | 0.0005 | 0.0240 |
| 0.020 | 0.0942 | 0.1477 | 0.0180 | 0.1477 |
| 0.050 | 0.9512 | 0.9649 | 0.7695 | 0.9649 |

With a 5% fault the naive revenue effect averages 61.45 INR against a true 29.40 INR (200 experiments).

### API latency

Measured with an in-process starlette TestClient, sequential requests, 2 threads; latency includes validation, JSON encoding and the analysis, not a network hop.

| call | payload | requests | p50 ms | p95 ms |
|---|---|---|---|---|
| POST /v1/experiments (decision report) | 40,000 rows, CUPED, 14 daily looks, SRM | 15 | 65.2 | 67.2 |
| POST /v1/targeting | 1,000 customers | 40 | 38.5 | 40.6 |
| POST /v1/observational/ate | 4,000 rows, 100 refit bootstrap resamples | 8 | 291.5 | 308.1 |

Timings (seconds): true_ate 1.05, ate_moderate_overlap 102.53, ate_weak_overlap 114.76, cate 18.13, cuped 6.63, sequential 12.92, srm 3.82, serving 8.46, total 268.31.
<!-- RESULTS-MD:END -->

## 4. Findings and decisions made along the way

1. **Unbounded outcome models broke AIPW.** In the first pilot, the per-arm ridge regressions on the
   40-column polynomial basis extrapolated to absurd revenue predictions on the few customers with extreme
   covariates, and the AIPW refit-bootstrap intervals came out far too wide. Predictions are now clipped to
   the range of the outcomes each model was fitted on (a conditional mean of revenue cannot be negative or
   exceed the largest basket seen), which removed the extremes without touching the estimand.
2. **Cross-fitting costs efficiency with a small parametric basis, and its refit bootstrap is
   conservative.** Pilots that refitted one nuisance model at a time showed that the propensity refit inflates
   the spread of the cross-fitted AIPW bootstrap most: the second-order nuisance term is larger in a resample,
   which holds fewer distinct rows. I kept both AIPW forms and report them side by side rather than drop the
   awkward one. In the final run the cross-fitted refit bootstrap covers 0.986 against 0.952 for its Wald
   interval (moderate overlap), while full-sample AIPW is calibrated under both intervals.
3. **The DR-learner suffers from heavy-tailed pseudo-outcomes.** Revenue residuals divided by small
   propensities give pseudo-outcomes with extreme tails, so the DR-learner's final regression is noisy at
   20,000 training rows. Two remedies were tried in a pilot and rejected: a stricter propensity clip (0.05) and
   5-fold cross-fitting changed little. A 50,000-customer history helped every learner and is what the
   benchmark uses. Raising the minimum leaf size from 50 to 500 was applied to all learners at once, after a
   three-seed pilot on seeds the benchmark does not use; no learner was tuned on PEHE. The DR-learner still
   trails the S- and X-learners (PEHE / SD 0.850 against 0.658 and 0.683), and that is reported as measured.
4. **A documentation error caught by a closed-form test.** The first Qini docstring said `Q(k)` is the
   incremental outcome of treating the top `k`. A test against the closed form for standard normal effects
   showed the coefficient converging to `1 / (4 sqrt(pi))`, half the uplift-curve limit: the Qini curve counts
   only the treated customers in the top `k`. The docstrings and EVALUATION.md now say so, and the test pins
   both limits.
5. **A decision bug caught by the verdict tests.** With daily looks, the report first judged SRM on the
   sequential test alone. That test is valid under peeking but has less power, and a 5 percent logging fault
   in 60,000 users went unflagged in two of three seeds, so the report said `ship` on an inflated effect. The
   rule now fires when either the chi-square test on the latest counts or the sequential test is below
   `srm_alpha`. The benchmark measures the combined rule under daily re-runs: false alarms in 0.70 percent of
   healthy experiments (nominal 0.1 percent for a single check) and detection of the 5 percent fault in 96.5
   percent. The asymmetry is deliberate: a false alarm costs an investigation, a miss ships a biased result.
6. **A misleading futility message.** When the fixed-horizon sample size was reached but the confidence
   sequence still included the MDE, the report said "about 0 more users per arm". It now says the planned
   sample is reached and the interval still includes effects of the MDE size.
7. **A packaging bug caught by the container check.** `causa.cli` imported the benchmark at module level; the
   latency study imports Starlette's test client, which needs `httpx2`, a dev-only dependency, so `causa serve`
   crashed at import in the production image. The CLI now imports the benchmark only for `causa bench`, and
   `tests/test_cli.py` fails if importing the CLI and the app loads the test client.
8. **CPU discipline.** The first full test run used about six cores because gradient boosting's OpenMP pool
   defaults to every core. The test session and model training now cap thread pools at two, which also made
   the suite faster, and the benchmark's worker processes are pinned to one thread each.
9. **Selection noise in the serving model.** Holdout Qini picked the X-learner for the served model in both
   the fresh-clone server and the container, while the benchmark's five seeds rank the S-learner first on
   policy value. The two are close on every metric, and choosing by an observable criterion on one holdout is
   what a team can do; the model info endpoint shows every candidate's score.

## 5. Quality gate history

* `4d8c7f8`: 42 checks written from theory and 150 to 400 replication pilots, committed before any full run.
* `71175dc`: the first full run passed all 42.
* `af2660b`: five checks added together with the measurements they gate: the SRM decision rule's false alarm
  rate (at most 0.01) and detection of the 5 percent fault (at least 0.9), and p95 latency bounds for the
  three API calls. No existing bound was changed (the diff also reformats 0.90 as 0.9).
* Every later run, including the fresh-clone run in section 7, passed all 47. The 42 original checks read the
  same values in every full run, because the benchmark is deterministic given its seed.

## 6. Commands and output tails

<!-- TAILS:BEGIN -->
All three were run in a shell started with `env -i HOME=$HOME PATH=$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin`
from the repository root, at commit `2b6b3f9`, while two other builds were running on the machine (load average
near 12 at the end of the benchmark).

#### `make lint`

```
uv run ruff check src tests
All checks passed!
uv run ruff format --check src tests
53 files already formatted
```

#### `make test` (tail)

```
........................................................................ [ 30%]
........................................................................ [ 61%]
........................................................................ [ 91%]
....................                                                     [100%]
236 passed in 27.16s
```

#### `make bench` (complete output)

```
uv run python -m causa.bench --check --update-readme
causa benchmark (seed 42, 2 workers)
  true_ate: 1.05 s
  ate_moderate_overlap: 102.53 s
  ate_weak_overlap: 114.76 s
  cate: 18.13 s
  cuped: 6.63 s
  sequential: 12.92 s
  srm: 3.82 s
  serving: 8.46 s
wrote benchmarks/results.json and benchmarks/results.md
README results section updated
PASS ate.scenarios.moderate_overlap.estimators.difference_in_means.coverage = 0.0000 (want <= 0.05)
PASS ate.scenarios.moderate_overlap.estimators.ipw.coverage = 0.9380 (want >= 0.9 and <= 0.99)
PASS ate.scenarios.moderate_overlap.estimators.aipw.coverage = 0.9360 (want >= 0.9 and <= 0.99)
PASS ate.scenarios.moderate_overlap.estimators.aipw_wald.coverage = 0.9420 (want >= 0.9 and <= 0.99)
PASS ate.scenarios.moderate_overlap.estimators.aipw_crossfit_wald.coverage = 0.9520 (want >= 0.9 and <= 0.99)
PASS ate.scenarios.moderate_overlap.estimators.aipw_crossfit.coverage = 0.9860 (want >= 0.9)
PASS ate.scenarios.moderate_overlap.estimators.ipw.bias = -2.4854 (want >= -10.0 and <= 10.0)
PASS ate.scenarios.moderate_overlap.estimators.aipw.bias = -1.7712 (want >= -10.0 and <= 10.0)
PASS ate.scenarios.moderate_overlap.estimators.aipw_crossfit.bias = -4.7780 (want >= -10.0 and <= 10.0)
PASS ate.scenarios.moderate_overlap.estimators.aipw.rmse = 35.5423 (want <= 50.0)
PASS ate.scenarios.moderate_overlap.overlap_mean.max_smd_after = 0.0425 (want <= 0.1)
PASS ate.scenarios.weak_overlap.estimators.difference_in_means.coverage = 0.0000 (want <= 0.05)
PASS ate.scenarios.weak_overlap.estimators.aipw.coverage = 0.9340 (want >= 0.85)
PASS ate.scenarios.weak_overlap.estimators.aipw.bias = 3.6424 (want >= -15.0 and <= 15.0)
PASS ate.scenarios.weak_overlap.overlap_mean.share_outside_band = 0.2646 (want >= 0.1)
PASS cate.learners.s_learner.pehe_rel = 0.6577 (want <= 0.8)
PASS cate.learners.x_learner.pehe_rel = 0.6827 (want <= 0.8)
PASS cate.learners.dr_learner.pehe_rel = 0.8503 (want <= 0.95)
PASS cate.learners.s_learner.policy_value_rel = 0.9132 (want >= 0.78)
PASS cate.learners.x_learner.policy_value_rel = 0.8709 (want >= 0.78)
PASS cate.learners.dr_learner.policy_value_rel = 0.8006 (want >= 0.72)
PASS cate.learners.x_learner.qini_rel = 0.8687 (want >= 0.5)
PASS cate.learners.dr_learner.qini_rel = 0.7499 (want >= 0.5)
PASS cate.learners.historical_rule.policy_value_rel = 0.1374 (want <= 0.35)
PASS cuped.variance_reduction = 0.3420 (want >= 0.28)
PASS cuped.cuped.bias = 0.0251 (want >= -3.0 and <= 3.0)
PASS cuped.cuped.coverage = 0.9470 (want >= 0.92)
PASS cuped.difference_in_means.coverage = 0.9445 (want >= 0.92)
PASS cuped.cuped.power = 0.8935 (want >= 0.85)
PASS cuped.power_gap.difference_in_means = -0.0151 (want >= -0.05 and <= 0.05)
PASS cuped.power_gap.cuped = -0.0034 (want >= -0.05 and <= 0.05)
PASS sequential.aa.reject.msprt.rate = 0.0140 (want <= 0.05)
PASS sequential.aa.reject.msprt_cuped.rate = 0.0115 (want <= 0.05)
PASS sequential.aa.reject.naive.rate = 0.2875 (want >= 0.15)
PASS sequential.aa.reject.fixed.rate = 0.0565 (want >= 0.03 and <= 0.07)
PASS sequential.ab.reject.msprt_cuped.rate = 0.6610 (want >= 0.55)
PASS sequential.ab.reject.fixed.rate = 0.7320 (want >= 0.65)
PASS srm.by_drop.drop_0_000.sequential.rate = 0.0004 (want <= 0.002)
PASS srm.by_drop.drop_0_000.chi_square_final.rate = 0.0016 (want <= 0.003)
PASS srm.by_drop.drop_0_050.chi_square_final.rate = 0.9512 (want >= 0.9)
PASS srm.by_drop.drop_0_050.sequential.rate = 0.7695 (want >= 0.6)
PASS srm.fault_bias.mean_estimated_effect = 61.4513 (want >= 45.0)
PASS srm.by_drop.drop_0_000.decision_rule.rate = 0.0070 (want <= 0.01)
PASS srm.by_drop.drop_0_050.decision_rule.rate = 0.9649 (want >= 0.9)
PASS serving.experiment_report.p95_ms = 67.2456 (want <= 5000.0)
PASS serving.targeting.p95_ms = 40.6312 (want <= 2000.0)
PASS serving.observational_ate.p95_ms = 308.0924 (want <= 10000.0)
quality gate: PASS

real	4m29.851s
user	7m52.347s
sys	0m11.922s
```
<!-- TAILS:END -->

## 7. From-scratch verification

`git clone` of commit `f0ab2e5` into an empty directory, then, in a shell started with `env -i` (only `HOME`
and a minimal `PATH`): `make setup`, `make lint`, `make test`, `make bench`. All passed: 233 tests, all 47 gate
checks, 3 minutes 34 seconds for the benchmark. Its `results.json` was identical to the committed one apart
from timings, latencies and run metadata (checked by a script that strips those keys and compares the rest).
The only code change after that run is the CLI import fix of finding 7, which touches no benchmark path.

## 8. `make serve`

From the fresh clone, `make serve` trained the targeting model on 50,000 customers (5.5 seconds, from the
JSON startup log) and answered `/health` 7 seconds after launch on port 8773. Over HTTP: the dashboard and its
assets returned 200; `POST /v1/experiments/simulate` with an 8 percent lift returned `ship`; a CSV upload of
`data/experiment.csv` returned `keep_running`; `POST /v1/targeting` with 500 sample customers and a 20 percent
budget selected 100; an empty customer list returned 422; `/metrics` showed decisions by verdict and 500
customers scored; an `X-Request-ID: verify-1` header was echoed and appeared in the access log.

## 9. Container

`docker build` of the multi-stage image succeeded (714 MB). Started with `docker run -p 8774:8773`, it ran as
`uid=999(causa)`, trained its model in 6.2 seconds, answered `/health` with `model_loaded: true` after 8
seconds, served the dashboard and the benchmark, returned `ship` for a simulated 8 percent lift, and Docker
reported it `healthy`. The first build exposed finding 7. The verification image was removed afterwards. The
GitHub Actions workflow runs lint, the tests, the benchmark gate and this image check on every push.

## 10. Known gaps and limitations

* **Synthetic data only.** The generators encode my assumptions about purchase probabilities, baskets and
  spending rates. No public dataset has per-customer ground truth, so PEHE and policy value against the oracle
  can only be measured on synthetic worlds; a Qini-only benchmark on public uplift data is on the roadmap.
* **ATE coverage is studied for one nuisance class.** The refit bootstrap and the coverage study use the
  fixed polynomial basis. Gradient-boosted nuisances appear only inside the DR- and X-learners, and their
  intervals are not studied.
* **Conservative or lower-powered pieces are reported, not fixed.** The cross-fitted AIPW refit bootstrap
  over-covers; the mixture SPRT is conservative on revenue, so its A/B power (0.433 without CUPED) is well
  below the fixed-horizon test's (0.732); the sequential SRM test catches 77 percent of the 5 percent fault
  on its own.
* **The decision report assumes outcomes are known at enrolment.** Delayed outcomes, novelty effects and
  ratio metrics (revenue per session) are not handled; guardrails use plain difference in means.
* **No LLM layer.** None was added, so nothing here needs an
  API key.
* **Latency was measured in-process with one client.** There is no load or concurrency test.
* **CI has never run** (no remote by instruction); the same steps were run locally as described above.
