# Architecture

causa is one Python package (`src/causa`) with no services besides its own API process. The layers depend
only downward: generators and numerical code know nothing about HTTP, and the API only converts between
JSON and the analysis types.

## Modules

| layer | module | responsibility |
|---|---|---|
| data | `data/campaign.py` | customers from three latent factors, the confounded historical targeting rule, closed-form potential outcome means and CATE, the population ATE by Monte Carlo |
| | `data/experiments.py` | randomised A/B tests as vectorised `(replications, users)` arrays or one table; pre-period revenue, guardrails, SRM fault; closed-form effects |
| numerics | `features.py` | the one encoding of customers every model reads (`feature_matrix`), input validation, and `Basis` (intercept, standardised columns, squares and pairwise products of continuous columns) |
| | `linear.py` | weighted ridge and logistic regression for a batch of weight vectors in one solve |
| observational | `propensity.py` | `PropensityModel`, effective sample size, standardised mean differences, `overlap_report` with warnings |
| | `ate.py` | difference in means, outcome regression, Hajek IPW, AIPW (full-sample and cross-fitted), refit bootstrap and Wald intervals |
| | `learners.py` | S, T, X and DR meta-learners over a shared scikit-learn base learner |
| | `uplift.py` | Qini and uplift curves, Qini coefficient, AUUC, true gain curve, budgeted policy, true and IPW policy value |
| | `targeting.py` | trains all learners, keeps the one with the best holdout Qini, decides coupons, persists with joblib |
| experiments | `experiment/stats.py` | per-arm sufficient statistics, difference in means (Welch), CUPED, delta-method relative effect |
| | `experiment/sequential.py` | mixture SPRT p-values and confidence sequences, sequential SRM test |
| | `experiment/srm.py`, `experiment/power.py` | chi-square SRM test; power, sample size and minimum detectable effect |
| | `experiment/report.py` | validates a table, computes every statistic at each daily look, applies the decision rule |
| evaluation | `studies/ate.py`, `studies/cate.py`, `studies/experiments.py`, `studies/serving.py` | the Monte Carlo and replication studies |
| | `bench.py`, `render.py` | run the studies, write `benchmarks/results.json` and `results.md`, copy tables into the README, apply the gate |
| service | `api/app.py`, `api/schemas.py`, `api/service.py`, `api/store.py` | routes, Pydantic schemas, the service layer (limits, error mapping, metrics), the sqlite report store |
| | `api/middleware.py`, `api/security.py`, `jsonlog.py`, `config.py` | request ids, Prometheus metrics, JSON logs, API key check, settings from the environment |
| | `api/static/` | the dashboard (HTML, CSS, vanilla JS; no build step) |
| | `cli.py` | `causa generate`, `analyze`, `ate`, `train`, `serve`, `bench` |

## Data flow

**Experiment decision report** (`POST /v1/experiments`, `/csv`, `/simulate`, and `causa analyze`):

```
JSON columns or CSV ──> schema checks (sizes, fields, spec bounds) ──> pandas table
  ──> report._prepare: required columns, variant labels, finite numeric metrics, at least 20 users per arm,
      day column turned into a look index
  ──> per-arm sums of y, y^2, x, x^2, xy per day via bincount, then cumulative over days
  ──> CUPED and difference-in-means contrasts at every look
  ──> mixture SPRT (tau = MDE): always-valid p-value and confidence sequence; fixed-horizon numbers kept
  ──> SRM: chi-square on the latest counts and the sequential test over all looks
  ──> power at the MDE with the CUPED-adjusted SD; guardrails as non-inferiority checks on their own sequences
  ──> decision rule ──> DecisionReport ──> sqlite (report only, never raw rows) ──> response with id
```

The decision rule, in order: `invalid` if either SRM check is below `srm_alpha`; `do_not_ship` if a guardrail
fails or the primary metric is significantly worse; `ship` if it is significantly better and every guardrail
passes; `keep_running` if it is better but a guardrail is not cleared; `stop_no_effect` if the confidence
sequence lies inside plus or minus the MDE; otherwise `keep_running`, with the users per arm still needed.
A guardrail passes when the upper end of its harm interval is within its tolerated harm, fails when the harm
is significant and the margin cannot be confirmed, and is inconclusive otherwise.

**Targeting** (`POST /v1/targeting`): customers ──> Pydantic bounds ──> `feature_matrix` ──> the serving
learner's predicted uplift ──> `budget_policy` (top share by uplift, only above the contact cost) ──> per
customer uplift, decision and rank, plus the expected incremental revenue of the selection.

**Observational ATE** (`POST /v1/observational/ate`, `causa ate`): columns ──> finite numeric arrays (at most
20,000 rows and 12 covariates by default) ──> `Basis` ──> one batched propensity fit and two batched outcome
fits for the original sample and every bootstrap resample, plus the same inside each cross-fitting fold
──> every estimator's point estimate, refit bootstrap interval and, for AIPW, Wald interval ──> overlap and
balance report with warnings.

**Benchmark** (`make bench`): studies ──> `results.json` ──> `render.py` ──> `results.md` and the README block
──> `--check` against `thresholds.json`.

## Design notes

**Batched refits.** A bootstrap resample is a vector of multinomial counts `w`. For ridge,
`beta_w = (Z' W Z + lam P)^-1 Z' W y`, and `Z' W Z = sum_i w_i z_i z_i'`. Stacking `B` count vectors as the rows
of a matrix and multiplying by the `(n, d*d)` matrix of row-wise outer products gives all `B` Gram matrices in
one BLAS call. Logistic regression uses the same product inside each Newton step with weights
`w_i p_i (1 - p_i)`, starting the resamples from the original-sample solution. Row 0 of every weight matrix is
all ones, so point estimates and bootstrap replicates come from the same call. The outer products are cached
when they fit in 16 million numbers and built in bounded row chunks otherwise, so a large upload cannot
exhaust memory.

**Sufficient statistics.** The experiment layer never loops over users per look: six sums per arm per look
are enough for difference in means, CUPED and their variances, and the same functions run on scalars, on a
daily history and on `(replications, looks)` grids in the Monte Carlo studies.

**Determinism and resources.** Generators take explicit seeds; studies derive per-replication seeds with
`SeedSequence`, so the worker count never changes a result. The benchmark uses at most two worker processes,
each pinned to one BLAS and OpenMP thread; model training and the test session cap OpenMP at two threads.

**Persistence.** Decision reports go to `data/experiments.sqlite` (one connection per call, safe under the
server's thread pool). The targeting model goes to `artifacts/targeting_model.joblib`; `load_model` refuses a
file that does not hold a `TargetingModel`. Both paths come from `CAUSA_DATA_DIR` and `CAUSA_ARTIFACTS_DIR`.

**Observability and security.** Every response carries `X-Request-ID` (echoed only if it is a short token);
access logs are one JSON object per line; `/metrics` exposes request counts and latency histograms by route
template (never raw paths), decisions by verdict, customers scored and analyses run. When `APP_API_KEY` is
set, every route except health, metrics, the dashboard and the docs requires `X-API-Key`, compared in
constant time. Analysis errors become 422 responses with the specific reason; size limits give 413; a
missing model gives 503.

**Extending.** A new ATE estimator is a function of the design and the weight matrix returning one value per
row, added to `METHODS` and the renderer. A new learner implements `fit(X, t, y)` and `predict(X)` and is
registered in `make_learner`; the CATE study and the targeting selection pick it up.
