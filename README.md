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
make bench                                  # about three minutes: every number below, plus the quality gate
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
