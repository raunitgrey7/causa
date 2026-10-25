# causa benchmark results

Seed 42, generated 2026-10-25T05:52:19+00:00, commit `2b6b3f9`, Python 3.13.17, numpy 2.5.4, scikit-learn 1.9.1 on Darwin arm64, 2 worker processes. Total benchmark time 268.31 s.

## Observational ATE on a confounded campaign

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

## CATE learners and budgeted targeting

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

## CUPED variance reduction and power

2000 randomised experiments of 20,000 users, true revenue effect 78.40 INR. Measured variance reduction 0.342; mean squared pre/post correlation (the theoretical reduction) 0.349.

| estimator | bias | SD | mean SE | coverage of 95% interval | power (empirical) | power (formula) |
|---|---|---|---|---|---|---|
| difference_in_means | -0.33 | 30.99 | 30.14 | 0.945 | 0.724 | 0.739 |
| cuped | 0.03 | 25.13 | 24.31 | 0.947 | 0.893 | 0.897 |

## Peeking and sequential testing

2000 replications per scenario, 20,000 users analysed at 40 equally spaced looks, alpha 0.05; mixing SD tau = 78.4 INR (the planned effect). A/B true effect 78.40 INR.

| strategy | A/A false positive rate | A/B power |
|---|---|---|
| z-test at every look (peeking) | 0.2875 (0.0101) | 0.8540 (0.0079) |
| z-test once at the end | 0.0565 (0.0052) | 0.7320 (0.0099) |
| mixture SPRT, always-valid | 0.0140 (0.0026) | 0.4330 (0.0111) |
| mixture SPRT on CUPED-adjusted revenue | 0.0115 (0.0024) | 0.6610 (0.0106) |

Under A/B the CUPED mixture SPRT stops after 14,351 users on average (median 15,500) of the 20,000 available.

## Sample ratio mismatch

10,000 experiments of 100,000 users over 14 days per row, alarm threshold p < 0.001. The fault drops that share of non-converting treatment users from the logs.

| fault (share lost) | chi-square once at the end | chi-square checked daily | sequential SRM, daily | report rule (either, daily) |
|---|---|---|---|---|
| 0.000 | 0.0016 | 0.0070 | 0.0004 | 0.0070 |
| 0.005 | 0.0026 | 0.0116 | 0.0005 | 0.0116 |
| 0.010 | 0.0092 | 0.0240 | 0.0005 | 0.0240 |
| 0.020 | 0.0942 | 0.1477 | 0.0180 | 0.1477 |
| 0.050 | 0.9512 | 0.9649 | 0.7695 | 0.9649 |

With a 5% fault the naive revenue effect averages 61.45 INR against a true 29.40 INR (200 experiments).

## API latency

Measured with an in-process starlette TestClient, sequential requests, 2 threads; latency includes validation, JSON encoding and the analysis, not a network hop.

| call | payload | requests | p50 ms | p95 ms |
|---|---|---|---|---|
| POST /v1/experiments (decision report) | 40,000 rows, CUPED, 14 daily looks, SRM | 15 | 65.2 | 67.2 |
| POST /v1/targeting | 1,000 customers | 40 | 38.5 | 40.6 |
| POST /v1/observational/ate | 4,000 rows, 100 refit bootstrap resamples | 8 | 291.5 | 308.1 |

Timings (seconds): true_ate 1.05, ate_moderate_overlap 102.53, ate_weak_overlap 114.76, cate 18.13, cuped 6.63, sequential 12.92, srm 3.82, serving 8.46, total 268.31.
