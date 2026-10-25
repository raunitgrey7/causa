# causa

Causal inference and experimentation for growth teams. causa answers the two questions a growth team asks
every week, and measures how well it answers them against synthetic worlds whose true effects are known:

1. **Did last quarter's coupon campaign work, and who should get the next one?** Marketers chose who got
   the coupon, so the data are confounded. causa models the propensity, checks overlap and balance,
   estimates the average effect by IPW and doubly robust AIPW with refit bootstrap intervals, fits S, T, X
   and DR learners for customer-level effects, and selects customers under a coupon budget.
2. **Should we ship the variant of the A/B test we have been watching every day?** causa uploads the
   experiment table and returns a decision report: a CUPED-adjusted effect, an always-valid confidence
   sequence that tolerates daily peeking, power at the planned minimum detectable effect, guardrail
   verdicts and sample ratio mismatch (SRM) checks, ending in one of five documented verdicts.

Everything runs on a laptop CPU with no API key and no network. The estimators are written on numpy and
scikit-learn, without causal-inference libraries, so each can be checked against a closed form. Every
number in the Results section was produced by `make bench` with seed 42 and copied from
`benchmarks/results.json`; `tests/test_docs_consistency.py` fails if the two disagree.

## Why

Most causal and experimentation code stops at a point estimate. The failure modes that cost growth teams
money are elsewhere, and each one is measured here instead of asserted:

* **Confounded campaign analysis.** Comparing coupon recipients with everyone else credits the coupon with
  the engagement that got customers targeted. The benchmark shows by how much, and whether the adjusted
  estimators' intervals actually cover the truth at their nominal rate.
* **Targeting by the wrong signal.** Sending coupons to the customers most likely to buy pays discounts to
  people who would buy anyway. The benchmark compares uplift targeting with the historical rule and with
  an oracle that knows every customer's true effect.
* **Peeking.** A z-test checked daily declares winners far more often than 5 percent of the time when
  nothing changed. The report uses always-valid inference, and the benchmark measures both error rates.
* **Broken randomisation.** A logging fault that loses some treatment users makes a flat change look like
  a big win. The benchmark measures how much the fault inflates the effect and how often the SRM checks
  catch it.

## Architecture

```
 seeded generators with ground truth                                 analysis
 ----------------------------------                                  --------
 data/campaign.py      confounded coupon campaign,  ---> features.py ---> linear.py: batched weighted ridge and
                       closed-form CATE per customer      Basis            logistic solves (one BLAS call for all
                                                                           bootstrap resamples)
                                                                              |
                                                     propensity.py  overlap, balance (SMD), effective sample size
                                                     ate.py         difference in means, outcome regression,
                                                                    IPW, AIPW, cross-fitted AIPW, refit bootstrap
                                                     learners.py    S, T, X, DR learners (gradient boosting)
                                                     uplift.py      Qini, AUUC, budgeted policy and its value
                                                     targeting.py   learner chosen by Qini on a randomised holdout

 data/experiments.py   randomised A/B tests with     ---> experiment/stats.py   per-arm sufficient statistics,
                       pre-period revenue, guardrails,                          difference in means, CUPED
                       SRM fault injection                 experiment/sequential.py  mixture SPRT, sequential SRM
                                                           experiment/srm.py, power.py
                                                           experiment/report.py  decision rule

 studies/ + bench.py   Monte Carlo against the ground truth -> benchmarks/results.json, results.md, quality gate
 api/                  FastAPI: /v1/experiments, /v1/targeting, /v1/observational/ate, /health, /metrics, dashboard
 cli.py                causa generate | analyze | ate | train | serve | bench
```

Module responsibilities and the request flow: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Generators,
estimands, metric definitions and how to reproduce: [docs/EVALUATION.md](docs/EVALUATION.md). Every
measured number with its command, and what went wrong on the way: [REPORT.md](REPORT.md).

## Quickstart

```
make setup                                  # uv sync, then sample CSVs in data/
make serve                                  # trains the targeting model on first start, serves http://127.0.0.1:8773
uv run causa analyze data/experiment.csv    # the decision report from the command line
make bench                                  # 3 to 5 minutes: every number below, plus the quality gate
```

Open `http://127.0.0.1:8773/` for the dashboard (simulate or upload an experiment, see the verdict and the
daily confidence sequence, score customers under a budget, read the benchmark), or `/docs` for the OpenAPI
UI. `make test` runs the test suite and `make lint` runs ruff. Set `APP_API_KEY` to require an `X-API-Key`
header on every route except `/health`, `/metrics`, the dashboard and the docs.

```
curl -s -X POST localhost:8773/v1/experiments/simulate -H 'content-type: application/json' \
     -d '{"n_users": 60000, "lift": 0.08}' | python -m json.tool | head -20
```

## Results

<!-- RESULTS:BEGIN -->
Seed 42, generated 2026-10-25T05:52:19+00:00, commit `2b6b3f9`, Python 3.13.17, numpy 2.5.4, scikit-learn 1.9.1 on Darwin arm64, 2 worker processes. Total benchmark time 268.31 s.

- **Confounding.** The naive difference in means is off by 165.2 INR against a true ATE of 20.62 INR. AIPW's bias is -1.77 INR (Monte Carlo SE 1.59) and its refit-bootstrap 95% interval covers the truth in 93.6% of 500 replications.
- **Weak overlap.** The diagnostics put 26.5% of customers outside [0.05, 0.95]; IPW's bias grows to 9.99 INR while AIPW's is 3.64 INR.
- **Targeting.** With coupons for 20% of customers, the s_learner earns 91.3% of the oracle's incremental revenue; the historical targeting rule earns 13.7%.
- **CUPED.** The pre-period covariate removes 34.2% of the variance (theory: 34.9%) and lifts power from 72.4% to 89.3%.
- **Peeking.** A z-test checked at all 40 looks declares a winner in 28.7% of A/A tests; the mixture SPRT does so in 1.4%.
- **SRM.** A logging fault that loses 5% of non-converting treatment users inflates the revenue effect from 29.40 to 61.45 INR; the report's SRM rule catches it in 96.5% of experiments and raises a false alarm in 0.70% of healthy ones.

### Observational ATE: bias and interval coverage against the true ATE

True population ATE 20.62 INR per customer (Monte Carlo over 4,000,000 customers, standard error 0.064); the CATE has SD 128.1 INR and is negative for 46.1% of customers. Bias and coverage are followed by their Monte Carlo standard errors in brackets.

#### moderate_overlap (targeting strength 1.0): 500 replications of 4000 customers, 200 refit bootstrap resamples each

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

#### weak_overlap (targeting strength 1.8): 500 replications of 4000 customers, 200 refit bootstrap resamples each

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

### CATE learners: PEHE, Qini, AUUC and policy value against the oracle

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

### CUPED

2000 randomised experiments of 20,000 users, true revenue effect 78.40 INR. Measured variance reduction 0.342; mean squared pre/post correlation (the theoretical reduction) 0.349.

| estimator | bias | SD | mean SE | coverage of 95% interval | power (empirical) | power (formula) |
|---|---|---|---|---|---|---|
| difference_in_means | -0.33 | 30.99 | 30.14 | 0.945 | 0.724 | 0.739 |
| cuped | 0.03 | 25.13 | 24.31 | 0.947 | 0.893 | 0.897 |

### Peeking against the sequential test

2000 replications per scenario, 20,000 users analysed at 40 equally spaced looks, alpha 0.05; mixing SD tau = 78.4 INR (the planned effect). A/B true effect 78.40 INR.

| strategy | A/A false positive rate | A/B power |
|---|---|---|
| z-test at every look (peeking) | 0.2875 (0.0101) | 0.8540 (0.0079) |
| z-test once at the end | 0.0565 (0.0052) | 0.7320 (0.0099) |
| mixture SPRT, always-valid | 0.0140 (0.0026) | 0.4330 (0.0111) |
| mixture SPRT on CUPED-adjusted revenue | 0.0115 (0.0024) | 0.6610 (0.0106) |

Under A/B the CUPED mixture SPRT stops after 14,351 users on average (median 15,500) of the 20,000 available.

### Sample ratio mismatch detection

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

Full tables including the value-by-budget curve: [benchmarks/results.md](benchmarks/results.md).
<!-- RESULTS:END -->

## API

| method and path | what it does |
|---|---|
| `POST /v1/experiments` | decision report for an experiment in columnar JSON (`{"spec": {...}, "data": {"variant": [...], ...}}`); stored, returns an id |
| `POST /v1/experiments/csv` | the same for a CSV upload (multipart `file`, optional JSON `spec` field) |
| `POST /v1/experiments/simulate` | generate an experiment with a chosen lift, logging fault or guardrail harm, and analyse it |
| `GET /v1/experiments`, `GET /v1/experiments/{id}` | stored reports (reports only; raw rows are never stored) |
| `POST /v1/targeting` | predicted uplift for each customer and coupon decisions under a budget and contact cost |
| `GET /v1/targeting/model` | the serving learner, every candidate's holdout Qini, and the IPW value-by-budget curve |
| `GET /v1/customers/sample` | synthetic customers in the scoring schema, for demos |
| `POST /v1/observational/ate` | overlap diagnostics and every ATE estimator with refit bootstrap intervals |
| `GET /v1/benchmark` | `benchmarks/results.json` |
| `GET /health`, `GET /metrics` | liveness and Prometheus metrics (requests, latency, decisions, customers scored) |

Requests are validated at the edge (finite numbers, bounded sizes, known fields); errors carry the
request id that also appears in the JSON access log and the `X-Request-ID` response header.

The experiment spec names the columns (`variant`, `day`, `revenue`, `pre_revenue` by default), the planned
split, the minimum detectable effect (absolute `mde`, or `mde_relative` to the control mean, 3 percent by
default), `alpha`, and guardrails as `{"metric": "support_contact", "max_harm": 0.01}`. The verdict is one of
`invalid`, `do_not_ship`, `ship`, `keep_running` or `stop_no_effect`, decided in that order by the rule
documented in `causa/experiment/report.py` and [docs/EVALUATION.md](docs/EVALUATION.md).

## Design decisions

* **Every bootstrap resample refits every nuisance model.** Holding the propensity model fixed across
  resamples misstates the variance of IPW. Refitting by looping over scikit-learn fits would make a
  500-replication coverage study take hours, so `causa.linear` solves all resamples of a weighted ridge or
  logistic regression in one batched matrix product: a resample is a vector of multinomial counts, and the
  Gram matrices of all resamples share the same row-wise outer products. The tests check that integer
  weights equal duplicated rows and that the batched solve equals a loop of single fits.
* **Two AIPW forms, measured side by side.** With a fixed polynomial basis, AIPW needs no sample splitting;
  cross-fitting is what flexible learners need. The benchmark reports both, so the cost of cross-fitting
  with a small parametric model is a measurement rather than folklore.
* **The serving uplift model is chosen by Qini on a randomised holdout,** the criterion a team can compute
  without ground truth. PEHE is reported in the benchmark only because the synthetic world knows the truth.
* **The decision report assumes daily peeking** and decides on always-valid intervals. Its SRM rule fires on
  either the chi-square or the sequential test, because a missed mismatch ships a biased result while a false
  alarm only costs an investigation; the benchmark measures that rule's false alarm rate under daily re-runs.
* **Thresholds were pre-registered.** The quality gate in `benchmarks/thresholds.json` was committed before
  the first full benchmark run; checks added later are listed in REPORT.md.

## Limitations

* All benchmark data are synthetic. The generators encode assumptions (logistic purchase probabilities,
  gamma basket values, a gamma-Poisson spending rate); on real data the estimators could rank differently,
  and no public dataset offers the per-customer ground truth that PEHE needs.
* The ATE coverage study uses a fixed polynomial basis for the nuisance models. Gradient-boosted nuisances
  are used only inside the DR- and X-learners, and their intervals are not studied.
* The cross-fitted AIPW refit bootstrap is conservative (it over-covers), and the DR-learner trails the S- and
  X-learners on this heavy-tailed revenue outcome; both are reported as measured, not tuned away.
* The mixture SPRT uses a plug-in variance, so its guarantee is asymptotic; on this revenue metric it is
  conservative, which costs power. The decision report treats each user's outcome as known at enrolment and
  does not model outcomes that accrue after the user arrives.
* Only mean metrics are supported: no ratio metrics (revenue per session) and no quantile metrics.
* API latency is measured in-process with one client; there is no load test.

## Roadmap

* Ratio metrics with the delta method, and CUPED for guardrails.
* Overlap-weighted and trimmed estimands for weak overlap, with their own ground truth.
* R-learner and a policy-tree learner; a coverage study of AIPW with gradient-boosted nuisances.
* Delayed outcomes in the sequential analysis.
* A Qini-only benchmark on a public uplift dataset (for example the Hillstrom email data).
* An optional LLM narrative of the decision report behind a flag, never in the decision path.

## Layout

```
src/causa/          package: data/, experiment/, studies/, api/ (static/ holds the dashboard), cli.py, bench.py
tests/              unit, property-based, API and benchmark-consistency tests
benchmarks/         results.json, results.md, thresholds.json (the quality gate)
docs/               ARCHITECTURE.md, EVALUATION.md
```
