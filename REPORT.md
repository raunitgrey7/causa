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
GitHub Actions workflow has not run, because the repository has no remote (by instruction).

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
* **No LLM layer.** The charter allows an optional LLM enhancement; none was added, so nothing here needs an
  API key.
* **Latency was measured in-process with one client.** There is no load or concurrency test.
* **CI has never run** (no remote by instruction); the same steps were run locally as described above.
