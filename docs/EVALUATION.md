# Evaluation methodology

Everything below is produced by `python -m causa.bench` (`make bench`) with seed 42. The numbers live in
`benchmarks/results.json` and `benchmarks/results.md`; this document says what they mean and how they were
made. All data are synthetic, generated with known ground truth, because the quantities that matter most
(the true ATE, every customer's true effect, the value of a targeting policy) are never observed in real data.

## 1. The campaign world (`causa.data.campaign`)

**Customers.** Three independent standard normal latent factors, engagement `z_e`, customer value `z_v` and
price sensitivity `z_p`, drive ten CRM covariates, which gives them the correlation real CRM data has (people
who order often also browse and open email more):

| covariate | generated as |
|---|---|
| `tenure_months` | 1 + Gamma(1.6, 18), capped at 120 |
| `orders_12m` | Poisson(exp(0.8 + 0.6 z_e + 0.15 z_v)) |
| `recency_days` | 1 + Exponential(300 / (1 + orders)), rounded, capped at 540 |
| `sessions_30d` | Poisson(exp(1.3 + 0.75 z_e)) |
| `email_open_rate` | logistic(-0.9 + 0.7 z_e + 0.6 e1) |
| `avg_order_value` | exp(7 + 0.4 z_v + 0.25 e2) times a city-tier factor (1.15, 1.0, 0.85) |
| `discount_affinity` | logistic(-1 + 1.1 z_p + 0.4 e3), the share of past orders that used a coupon |
| `app_user`, `loyalty_member` | Bernoulli, driven by engagement, and by tenure and value |
| `city_tier` | 1, 2 or 3 with probabilities 0.45, 0.35, 0.20 |

**Historical targeting (the confounder).** Marketers sent the coupon with probability
`e(x) = logistic(s * g(x))`, where `g` rises with sessions, email open rate, recency and order value (win-back
of lapsed high-value customers, including a recency times value term) and falls for loyalty members. `s` is
the targeting strength: 1.0 in the main scenario (3.8 percent of customers outside [0.05, 0.95] on average)
and 1.8 in the weak-overlap scenario. Because `g` is linear in standardised signals, their squares and one
product, a logistic model on the package's polynomial basis is correctly specified for the propensity.

**Potential outcomes.** Over the 30 days after the campaign a customer buys with probability
`p_0(x) = logistic(b(x))` without the coupon and `p_1(x) = logistic(b(x) + delta(x))` with it, and spends a
basket of expected value `v_0(x)` (the average order value, plus 15 percent for loyalty members) or
`v_1(x) = v_0(x) * (1 - 0.15) * (1 + 0.10)`: the coupon cuts the price by 15 percent and lifts basket size by
10 percent. Expected revenue is `mu_t(x) = p_t(x) v_t(x)` and the CATE is `tau(x) = mu_1(x) - mu_0(x)`, known
in closed form for every customer. `delta(x)` encodes four uplift segments: price-sensitive persuadables
(positive), lapsed customers who respond to a win-back (positive), sure things who would buy anyway and now
pay 6.5 percent less (negative revenue effect), and sleeping dogs, frequent buyers who rarely open email and
are put off by another promotion (negative). The benchmark reports the share of customers with a negative
effect next to the true ATE (it is close to half), so targeting matters. Realised revenue is a purchase indicator times a gamma basket draw (shape 2.5), so it is
zero for about two thirds of customers and right skewed otherwise; both potential outcomes share the purchase
uniform and the basket draw.

**Randomised holdouts.** The same generator with `randomized=True` assigns the coupon with probability 0.5;
the CATE study scores learners on such holdouts, as a team would.

## 2. The experiment world (`causa.data.experiments`)

A checkout redesign is tested 50/50 for 14 days. Each user has a gamma spending rate `lambda` (shape 0.6,
mean 1.5 orders a month) and a lognormal typical basket (mean 1400 INR). Orders are Poisson: over the 28
pre-period days at rate `lambda`, and over the 14-day window at `lambda * (1 + lift)` in treatment. Order
values are gamma (shape 4) around the user's basket, so revenue is zero-inflated and skewed, and the shared
rate makes the pre-period revenue a useful CUPED covariate (squared correlation about 0.35). Closed forms
(`expected_effects`): control mean revenue `1.5 * 14/30 * 1400 = 980` INR, revenue effect `980 * lift`, and
conversion `1 - (1 + s)^(-0.6)` from the gamma Laplace transform, with `s` the scaled rate. Guardrails are
`support_contact` (Bernoulli 4 percent, plus an optional harm in treatment) and `page_load_ms` (lognormal
around 800 ms, plus an optional shift). The SRM fault drops a chosen share of non-converting treatment users
from the logs, which both unbalances the arms and inflates the treatment mean.

## 3. Estimands

* **Population ATE** `E[tau(X)]`, computed by Monte Carlo over 4,000,000 fresh customers from a dedicated
  seed stream; its Monte Carlo standard error is reported next to it (0.064 INR in the published run). The
  targeting strength moves assignment only, so both overlap scenarios share this estimand. Interval coverage
  is coverage of this population quantity.
* **CATE** `tau(x)` for each holdout customer.
* **Policy value**: the true incremental revenue per customer of treating the set `S`,
  `mean(1[i in S] * (tau_i - cost))`. The oracle policy treats the top `budget` share by true `tau` (only
  where `tau > cost`).
* **Experiment effect**: the closed-form difference in mean window revenue between arms.

## 4. Metrics

| metric | definition |
|---|---|
| bias | mean over replications of (estimate - truth); its Monte Carlo SE is SD / sqrt(R) |
| SD, RMSE | standard deviation of the estimates; root mean squared error against the truth |
| coverage | share of replications whose 95% interval contains the truth; Monte Carlo SE sqrt(c (1 - c) / R) |
| mean width | mean interval width |
| PEHE | sqrt(mean((tau_hat - tau)^2)) on the holdout; also divided by SD(tau), where 1.0 equals predicting a constant |
| Qini curve | after sorting the holdout by score, `Q(k) = Y_t(k) - Y_c(k) N_t(k) / N_c(k)` over the top `k`, divided by the holdout size: the incremental outcome among the treated customers in the top `k` |
| Qini coefficient | area between the Qini curve and the straight line of random targeting; zero means no better than random. Reported relative to ranking the same holdout by the true CATE |
| uplift curve, AUUC | `U(k) = (Y_t(k)/N_t(k) - Y_c(k)/N_c(k)) * k / n`, the estimated incremental outcome of treating all of the top `k`; AUUC is its area |
| true gain area | area under the cumulative true CATE in score order, relative to the oracle ordering: the noise-free counterpart of AUUC |
| policy value / oracle | true value of the budgeted policy built from the scores, over the oracle policy's value |
| IPW policy value | `mean(1[i in S] * (t_i y_i / 0.5 - (1 - t_i) y_i / 0.5))` on the randomised holdout: what a team can estimate without ground truth |
| variance reduction | `1 - Var(CUPED estimates) / Var(difference-in-means estimates)` across replications, compared with the mean squared pre/post correlation |
| power | share of A/B replications with a significant result; the formula column is the normal-approximation power from `causa.experiment.power` with the measured SD |
| false positive rate | share of A/A replications declared significant at any look |
| SRM alarm rate | share of experiments flagged at p < 0.001 |

The Qini and AUUC definitions were checked against closed forms: for standard normal effects with a 50/50
split, a perfect ranking gives AUUC `1 / (2 sqrt(pi)) = 0.282` and a Qini coefficient of half that
(`tests/test_uplift.py`).

## 5. Studies

| study | design | replications |
|---|---|---|
| observational ATE | 4,000 customers per replication, two overlap scenarios, seven estimator and interval pairs, 200 refit bootstrap resamples each | 500 per scenario |
| CATE and targeting | train on 50,000 confounded customers, score on a 40,000-customer randomised holdout, budget 20 percent, contact cost 0 | 5 seeds |
| CUPED | 20,000 users, true lift 8 percent | 2,000 |
| peeking | 20,000 users analysed at 40 evenly spaced looks, A/A and A/B (8 percent lift), mixing SD equal to the planned effect | 2,000 per scenario |
| SRM | 100,000 users over 14 daily looks, faults losing 0 to 5 percent of non-converting treatment users | 10,000 per fault size (200 user-level runs for the bias) |
| API latency | in-process client: 40,000-row decision reports, 1,000-customer targeting calls, 4,000-row ATE analyses | 15, 40 and 8 requests |

**Estimators in the ATE study.** Difference in means; outcome regression (per-arm ridge regressions on the
basis, predictions clipped to the fitted outcome range); Hajek IPW with the basis logistic propensity clipped
to [0.01, 0.99]; AIPW with nuisances fitted on the full sample; and AIPW with 2-fold cross-fitting. The basis
is an intercept, the standardised features, squares and all pairwise products of the continuous ones (40
columns). Every estimator gets a percentile interval from 200 nonparametric bootstrap resamples that refit
every nuisance model; both AIPW forms also get the Wald interval from their influence function.

**Learners in the CATE study.** S, T, X and DR learners share one base learner, scikit-learn's histogram
gradient boosting (learning rate 0.05, up to 400 trees, 15 leaves, at least 500 rows per leaf, L2 1.0, early
stopping on a 10 percent split). The leaf size was raised from 50 to 500 for every learner at once after a
three-seed pilot, because 50 let all of them overfit noisy revenue; no learner was tuned on its own and none
on PEHE. The X- and DR-learners use the basis logistic propensity model.

## 6. Seeds and determinism

Each replication draws its seeds from `numpy.random.SeedSequence([42, scenario, replication])`, so results
do not depend on the number of worker processes or the order in which they finish.
`tests/test_bench.py` runs the quick configuration with one and with two workers and requires identical
numbers (only wall-clock timings and latencies may differ).

## 7. Quality gate

`benchmarks/thresholds.json` holds the bounds `python -m causa.bench --check` enforces in CI. The first 42
checks were committed before the first full run; their coverage bands are about three Monte Carlo standard
errors around what each interval should deliver, and the cross-fitted AIPW bootstrap only has a lower bound
because pilot runs showed it is conservative. Five checks (the SRM decision rule and API latency) were added
later together with the measurements they gate; REPORT.md lists them.

## 8. How to reproduce

```
make bench                                   # the published run, 3 to 5 minutes on two worker processes
uv run python -m causa.bench --quick --out build/quick   # seconds; small sizes, not the published numbers
uv run python -m causa.bench --no-run --check            # re-apply the gate to the committed results
```

`--workers` sets the number of worker processes (default 2); each is limited to one BLAS and OpenMP thread.
