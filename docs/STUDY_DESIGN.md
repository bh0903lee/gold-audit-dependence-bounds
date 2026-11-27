# Study designs

This document gives, for every study in the package, the generator, sample sizes, seeds, audit budgets, methods, reported quantities and random-number keys, in the order of the article's sections. Configuration files named below are those of this package.

Notation follows the article: `s` is the stay probability of the latent chain, `m` the latent prevalence, `B` the audit budget, `K` the calibration budget, `N` the number of observation episodes, `alpha` the total error probability, `theta = E[(2 X_0 - 1)(2 X_1 - 1)]` the target, and `f = P(X_1 = 0, Y_1 = 1)` the false-positive mass. M, N and F denote marginal, neighbor and full audit information.

---

## 1. Main simulation study (article Sections 6.1-6.4, Supplementary Sections S6 and S8)

Configuration: `configs/main.json` (seeds 91001-91200).

- Generator: stationary binary Markov latent process with stay probability 0.2, 0.5, 0.8; `Y_t = X_t D_t` with detection probability 0.95 or 0.20 according to the previous latent state. Independent episodes of length `L = 3` (and `L = 4` as a sensitivity analysis) are collected. In an audit episode only the central `X` is revealed; all `Y` are observed.
- Linear program: the word probabilities over the alphabet {(0,0), (1,0), (1,1)} are the variables (27 for `L = 3`); constraints are the observed word frequencies, stationarity, the audited `E[X_1]` (marginal audit), optionally `E[X_1 Y_0]` and `E[X_1 Y_2]` (neighbor audit), or the joint law of the audited `X_1` with the whole observed pattern (`2^L` constraints, full audit).
- Intervals: Hoeffding, two-sided empirical-Bernstein (Maurer and Pontil, Theorem 4, applied to each tail, `ln(4/delta)`) and Clopper-Pearson, each with a total `alpha = 0.05` allocated by a union bound over all empirical constraints. Sample-wise nesting of the intervals of different information sets is not guaranteed, because the allocation differs.
- Sample: 200 independent seeds x 3 stay values x 5 budgets (`B` in {50, 100, 500, 1000, 5000}), `N = 50,000`; paired width differences within a seed. The nominal 95% is the confidence-set guarantee of each configuration separately. Coverage is reported with Monte Carlo standard errors and Wilson intervals.
- Reported quantities: interval width for the true latent lag-one spin moment, coverage, containment of the population identified interval, frequency of empty confidence sets, and the exact population identified interval for each (stay, `L`, information set).
- Excess-width diagnostic (neighbor audit, `L = 3`, `B = 500`): the width with the observation moments fixed at their population values, the width with the audit moments fixed, and the width with both fixed (the population width).
- Empty confidence sets: when the program is infeasible the row is kept with the flag `empty_confidence_set = True` and the conservative interval [-1, 1]; coverage counting empty sets as misses is reported alongside. Solver failures are kept as status rows with the error and are never counted as intervals.

---

## 2. Six-variable comparison (Supplementary Sections S4 and S5)

Configuration: `configs/main_w3.json`.

Data. `L = 3`, `N = 50,000`, `B` in {50, 100, 500, 1000, 5000}, `s` in {0.2, 0.5, 0.8}, detector (0.95, 0.20), `alpha = 0.05`.
(A) The arrays of the main study (seeds 91001-91200), analyzed separately (`reanalysis_v42`).
(B) Seeds 92001-92200, the width-comparison seeds (`width_comparison`).
(C) Seeds 92201-93000 with the same configuration, combined with (B) for 1000-seed coverage and containment estimates (`extended`). (A) and (B) are not mixed and are tabulated separately. Random-number consumption for new seeds: `default_rng(seed)`, then stay order, then the observation cohort, then the audit cohorts in budget order.

Methods (all on the same samples, total `alpha = 0.05`, the same information: observed `Y` triples and the audited `X_1, Y_0, Y_2`).

- Neighbor word program, Hoeffding / empirical-Bernstein / Clopper-Pearson, `M = 11`, `delta = alpha/11`.
- Six-variable program, Clopper-Pearson, five intervals each at `delta = alpha/5`; `m, a, b` from the audit sample (`X_1`, `Y_0 X_1`, `X_1 Y_2`); `y, h` from the observation cohort (`Y_1`, `Y_0 Y_1`; `Y_1 Y_2` is not used). Variables `(m, y, a, b, h, t)`, nonnegativity of the nine entries of `K(t)`, minimum and maximum of `1 - 4m + 4a + 4b - 4h + 4t`. If infeasible, fallback [-1, 1] with a flag.
- Six-variable program with Hoeffding and two-sided empirical-Bernstein (`ln(4/delta)`) intervals, `delta = alpha/5`.
- Percentile bootstrap of the closed-form endpoints: 2,000 multinomial resamples of the observation cohort (8 patterns) and of the audit cohort (5 cells); interval `[q_{alpha/2}(theta_L*), q_{1-alpha/2}(theta_U*)]` truncated to [-1, 1]; if the lower limit exceeds the upper, an empty interval (miss, width 0, flagged). No finite-sample guarantee.
- Posterior interval: Dirichlet(1/2) prior on the observation 8-cell and audit 5-cell multinomials; 4,000 posterior draws; closed-form endpoints; `[q_{alpha/2}(theta_L), q_{1-alpha/2}(theta_U)]`. Posterior draws incompatible by the compatibility corollary are discarded and their fraction reported; if fewer than 100 compatible draws remain, fallback [-1, 1] with a flag. No frequentist guarantee.
- Projected L1 regions (Weissman et al. 2003) for the observation 8-cell and the audit 5-cell distributions, each at `delta' = alpha/2` with radius `sqrt(2 ln((2^k - 2)/delta') / n)`, linearized with auxiliary variables and projected through the word program. Guaranteed (union of two regions).
- The internal random numbers of the bootstrap and the posterior interval use `default_rng([seed, round(1000 s), B, method number])`, separate from the sampling stream.

Reported quantities. Mean of the paired relative width reduction `100 (1 - W_cand / W_ref)` within seeds with a Student t 95% Monte Carlo interval; coverage of the true theta (Wilson 95%); fraction of samples containing the whole population identified interval (tolerance 1e-9); fallback and empty-interval fractions with denominators preserved.

---

## 3. Assumption-violation study (article Section 6.5 and Fig. 3, Supplementary Section S10)

Configurations: `configs/main_w2a.json` (1000 seeds), `configs/main_w2b.json` (200 seeds).

Common. `L = 3`, `N = 50,000`, `alpha = 0.05`, `s` in {0.2, 0.5, 0.8}; the baseline generator is that of the main study. One axis is changed at a time. Methods: neighbor word program with Clopper-Pearson intervals, the six-variable program with Clopper-Pearson intervals, and, for reference, the neighbor word program with Hoeffding intervals. The target theta is the true latent lag-one spin moment of the observation-cohort population (also under audit-cohort drift). Common random numbers `default_rng([seed, stay index, 2])` are shared across levels, so cells of different levels within a seed are not independent. At level 0 the simulators of axes V1, V2a, V2b, V4a, V4b and V5 reproduce the baseline random stream and arrays (unit test). Axes V3a, V3b and V6 have no level 0 or consume random numbers differently and are not analyzed as paired with the baseline.

| Axis | Violation | Levels (excluding 0) | Violates the theorem's assumptions |
|---|---|---|---|
| V1 | `Y_t = X_t D_t + (1 - X_t) F_t`, `P(F = 1) = phi` (both cohorts) | phi in {0.002, 0.005, 0.01, 0.02, 0.05, 0.10} | yes |
| V2a | audit cohort only: stay `s + Delta`, or detector `(0.95 - Delta_d, 0.20 + Delta_d)` | Delta in {0.05, 0.10, 0.20} (Delta_d = 0); Delta_d in {0.05, 0.10} (Delta = 0) | yes |
| V2b | at positions t = 0, 1, 2, `P(D_t = 1 | X_{t-1} = 0) = 0.95 - kappa t` (both cohorts) | kappa in {0.05, 0.10, 0.20} | yes |
| V3a | symmetric second-order Markov latent chain: stay `s + g(1 - s)` after a stay, `s - g s` after a transition | g in {0.25, 0.5, 0.75} | no (in-model) |
| V3b | detector depending on `X_{t-2}` (`P(D = 1 | X_{t-1}, X_{t-2}) = [[0.95, 0.70], [0.35, 0.05]]`) or on `D_{t-1}` (`P(D_t = 1 | X_{t-1}, D_{t-1}) = [[0.80, 0.99], [0.05, 0.50]]`) | 2 variants | no (in-model) |
| V4a | the revealed `X_1` is flipped with probability epsilon | epsilon in {0.01, 0.02, 0.05} | yes |
| V4b | recorded as 0 with probability epsilon only when `X_1 = 1, Y_1 = 0` (anchoring) | epsilon in {0.05, 0.10, 0.20} | yes |
| V5 | `P(audit | Y_1 = 1) / P(audit | Y_1 = 0) = rho` (rejection sampling to exactly B audits) | rho in {2, 4} | yes |
| V6 | N length-3 windows at spacing `step` from one long sequence (observation); B windows from an independent long sequence (audit) | step in {1, 3, 30} | yes (independent episodes) |

The true value under V3a is `theta = 2q - 1`, `q = p_cs / (1 - p_ss + p_cs)`.

Seeds and cells. 1000 seeds (92001-93000): axes V1, V2a, V4a, V4b, V5 and the baseline; `B` in {500, 5000}; word program and six-variable program, both Clopper-Pearson. 200 seeds (93001-93200): all axes and the baseline; `B` in {100, 500, 5000}; three methods. The two sets are not mixed. Seed numbers shared with the six-variable comparison are not repetitions of the same experiment because streams and conditions differ.

Reported quantities. (i) The contaminated population moments `(m*, y*, a*, b*, h*)` by exact enumeration, the closed-form interval `[theta_L*, theta_U*]`, and the population excess `d = max(theta_L* - theta, theta - theta_U*, 0)` (deterministic), together with the compatibility of the contaminated moments; (ii) finite-sample coverage of the true theta, with the fallback [-1, 1] counted as covering and, alongside, with empty sets counted as misses; (iii) width; (iv) fraction of empty confidence sets; (v) the one-sided refutation check: the one-sided model is rejected when the audit sample contains at least one record with `X_1 = 0, Y_1 = 1`; rejection rate (compared with the analytic `1 - (1 - p)^B`) and coverage conditional on passing the check. The levels at which `d` first becomes positive are also reported on a fine grid, independent of the simulated levels.

Unit-test seeds do not overlap the study seeds 91001-93200.

---

## 4. Bounded false positives (article Sections 4.4, 5.2 and 6.6, Supplementary Section S11, Table S8, Fig. S3)

Configuration: `extensions/configs/protocol.json`.

Linear program. Alphabet `Z = 0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1)` with explicit lookups for `X` and `Y`; `L = 3` (and `L = 4` as a secondary comparison); nonnegativity, normalization, prefix/suffix word-mass equality; `f(p) <= epsilon`; the target is the mean adjacent spin product, which under stationarity equals the first-pair target. Each information set receives only the observed `Y` law and the audit moments it permits. Information hierarchy: N (`m, a, b`), N6 (`m, a, b, j`), N7L (N6 plus `c_L`), N7R (N6 plus `c_R`), N8 (N6 plus `c_L, c_R`), F (full `H(Y_0, X_1, Y_1, Y_2)`), where `j = E[X_1 Y_1]`, `c_L = E[Y_0 X_1 Y_1]`, `c_R = E[X_1 Y_1 Y_2]`; M8 is the pair formulation on the eight moments `(m, y, a, b, h, j, c_L, c_R)` without the observed triple law. The gluing certificates check the pair matrix `K >= 0` with equal row and column marginals, the coarsening identities, the triple construction `T[i,z,k] = H[g(i),z,g(k)] K[i,z]/H_L[g(i),z] K[z,k]/H_R[z,g(k)]` with zero terms where denominators vanish, nonnegativity, both pair marginals, recovery of `H`, agreement of the objectives, the pair-state chain and attainability of the whole interval, in exact rational arithmetic with stored `K, T, H`.

Population study (`d1`). Law banks: (A) the baseline Markov detector laws with `s` in {0.2, 0.5, 0.8}, detectors (0.95, 0.20) and (0.80, 0.80), `phi = 0`, at `L = 3` and `L = 4`, plus 128 ternary periodic mixtures (seeds 300001-300128); (B) two-sided parametric grid: `s` in {0.2, 0.5, 0.8}, the same two detectors, `phi` in {0.002, 0.01, 0.05, 0.10}, 24 laws; (C) general stationary rational bank: 512 laws (seeds 310001-310512), each a mixture of three cyclic words with lengths drawn uniformly from {3, ..., 8}, symbols drawn uniformly from the four states, integer mixture weights from {1, ..., 5} normalized, uniform phase; laws with `f = 0` or degenerate laws are tagged, not excluded. For each law with positive `f`, `epsilon` in {`f`, min(1, `f` + 0.005), min(1, `f` + 0.02), min(1, `f` + 0.10)}, duplicates computed once; the same increments for `f = 0`. All of N, N6, N7L, N7R, N8, F and the M8 pair program are computed on each law; the `L = 4` comparison uses the first 64 laws of bank C, with F holding the audit cross masses of `X_1` with all 16 `Y_0:Y_3` patterns. Reported: endpoints, width, `gap_lower = L_F - L_N6`, `gap_upper = U_N6 - U_F`, `gap_max`; maximum endpoint error of N8 against F, closed-form error at `epsilon = 0`, maximum nesting violation. A gap is counted when the floating-point gap exceeds 1e-6; the stored exact rational certificate is for the smallest seed with such a gap and, within the seed, the smallest `epsilon`. Tolerances: numerical equivalence 1e-8; primal residuals above 1e-7 are solver-invalid; one retry without presolve; persistent failures are stored with their status and kept in the denominator.

Finite-sample study (`d2`). `L = 3`, `N = 10,000` independent episodes; total gold budget `T` in {200, 1000, 4000}, split into `B = T/2` moment-audit episodes and `K = T/2` calibration episodes generated independently from the same population; only the central `X_1` is provided to the inference. Primary family: `s` in {0.2, 0.5, 0.8} x `phi` in {0, 0.02, 0.10}, detector (0.95, 0.20), `T = 1000`, nine cells, seeds 410001-411000 (1000 replicates per cell). Further cells: the same nine cells at `T = 200` and `T = 4000`, seeds 420001-420200; prevalence check with `m` in {0.2, 0.8}, lag persistence `rho = 0.6` with `P01 = m(1 - rho)`, `P10 = (1 - m)(1 - rho)`, detector (0.95, 0.20), `phi = 0.10`, `T = 1000`, seeds 430001-430200; the true joint `f = (1 - m) phi` and `theta` are computed by independent enumeration. Random numbers: `SeedSequence([20261014, 11, 2, condition_id, seed, cohort_id])`, cohorts 0 (observation), 1 (moment audit), 2 (calibration), 9 (analysis bootstrap). Methods on the same samples: `naive_zero` (three-state neighbor program, `epsilon = 0`, Clopper-Pearson `alpha = 0.05`; no guarantee when false positives are present); `oracle_joint` (four-state neighbor program, `epsilon` = true `f`, moment error 0.05; not available in practice); `learned_upper` (four-state neighbor program, `epsilon = U_CP(k, K; 0.99)`, moment error 0.04 split over the 8 patterns and 3 audit moments, calibration error 0.01; `U > 0` when `k = 0`, `U = 1` when `k = K`); `plugin_joint` (`epsilon = k/K`, moment error 0.05; ignores the estimation error of `epsilon`). Reported per replicate: `contains_population` (the sampled outer interval contains the whole identified interval of the true population under N information at `epsilon` = true `f`, tolerance 1e-8), `covers_truth`, width (domain [-1, 1]), empty set, solver failure, fallback used, whether `epsilon` covers `f`, `U - f`, width minus oracle width. Empty confidence sets and solver failures are never merged with successes; solver failures are counted as misses and receive the reporting fallback [-1, 1]. Monte Carlo intervals: for `contains_population` of `learned_upper` in the nine primary cells, two-sided Clopper-Pearson intervals at level `1 - 0.05/9` (summary key `coverage_check`: `large_loss_excluded` when all nine lower limits are at least 0.93, `refutation_signal` when any upper limit is below 0.95); width excess relative to the oracle: seed-paired means with 10,000 paired-bootstrap 95% intervals. Summary key `precision_summary` is `nonvacuous_tested_cells` when, in at least three of the six primary cells with nonzero false positives, the fallback-inclusive mean width is below 1.8 and the fallback rate below 0.05. Nesting diagnostic: for the first 50 seeds of each primary cell, one master Clopper-Pearson box over the 8 observation and 16 audit cells (error 0.04/24) with the same calibration limit; F uses the master constraints and N, N6, N8 receive only outer aggregations of the same cell limits, so that F is nested in N8, N6 and N; this is an implementation check of information nesting.

---

## 5. General stationary laws (article Section 6.2, Supplementary Section S7, Table S6, Fig. S2)

Configuration: `extensions/configs/addendum_e7_e9.json`, key `e9`.

Laws: without false positives, the 40 ternary periodic mixtures of bank A with seeds 300001-300040; with false positives, bank C seeds 310041-310080 (40 laws), analyzed with the learned upper limit. Degenerate laws (zero population width, or `f = 0` in bank C) are tagged and not excluded.

Samples. Independent episodes, `N = 10,000`; `B` in {200, 500, 2000}; `K = B` for laws with false positives; 200 replicates per law and budget, seeds 450001-450200; random numbers `SeedSequence([20261014, 11, 4, condition_id, seed, cohort_id])` with `condition_id = 3 * law_index + budget_index`, cohorts 0 (observation), 1 (audit), 2 (calibration), 9 (analysis); laws are converted exactly to probability vectors and words are sampled independently; M, N and F share the samples and `epsilon`. Clopper-Pearson intervals with the moment error split equally over the 9, 11 and 24 constraints of M, N and F; bank A with `epsilon = 0` and moment error 0.05; bank C always with the learned upper limit (`K = B`, moment error 0.04, calibration error 0.01) even when the true `f` is 0. `contains_population` for each method is containment of that method's own population identified interval (computed with the method's information and the true `f`, tolerance 1e-8); the common point-target coverage is reported separately. Solver failures are misses, empty sets raw misses, both with reported width 2 and fallback flag. Relative width reduction is the ratio of mean widths, `1 - mean(W_strong) / mean(W_weak)`, with fallback widths 2 included; a zero denominator is reported with its reason and the law is not deleted; strictness tolerance 1e-8.

Reported: for every law and budget, the Monte Carlo 95% Clopper-Pearson interval of N's `contains_population` (laws whose lower limit is below 0.93 are listed in `containment_lower_below_093`); at `B = 500`, the number of laws with `mean W_N < mean W_M` (`narrower_laws` of `total_laws`); the median and the 10th and 90th percentiles of the relative reduction (linear quantile method) for all laws and by bank.

---

## 6. Paired comparison of the learned and the true false-positive budget (Supplementary Section S7)

Configuration: `paired_budget/configs/p2.json`.

Laws: the 40 bank C laws (seeds 310041-310080); the true `f*` is 0 for two of them. Arms: `oracle_f_05` (`epsilon = f*`, moment error 0.05), `oracle_f_04` (`epsilon = f*`, moment error 0.04), `learned` (`epsilon` = calibration Clopper-Pearson upper limit, moment error 0.04 plus calibration error 0.01), `plugin` (`epsilon = k/K`, moment error 0.05; no guarantee). `known_zero` (`epsilon = 0`, moment error 0.05) is run only on the two laws with `f* = 0`. Reference: for each law and information set, the identified interval at the true `f*` and the true moments, and the true `theta`. Samples: `N = 10,000`, `B` in {200, 500, 2000}, `K = B`, 200 replicates, seeds 480001-480200; the four arms share the observation, audit and calibration samples (`SeedSequence([20261014, 11, 8, condition_id, seed, cohort])`, `condition_id = 3 * zero_based_law_index + budget_index`). Statistic: for law l, arm a and budget B, `g_{l,a} = 1 - mean(W_N) / mean(W_M)` over the 200 replicates with fallback width 2 included (the definition of Section 5 of this document); effect `Delta_l = g_{l,oracle_f_05} - g_{l,learned}` (paired within law); reported at `B = 500` as the median of `Delta_l` over the 40 laws. Monte Carlo uncertainty: 2,000 paired percentile bootstrap resamples of the within-law replicate index, applied jointly to the four arms and to M, N and F (the 95% interval of the median); resampling of laws is a separate sensitivity analysis; `np.quantile` with the linear method; the intervals are approximate Monte Carlo uncertainty. `oracle_f_05 - oracle_f_04` and `oracle_f_04 - learned` are reported separately, and the bank C median under `oracle_f_05` is set beside the stored bank A value (`bank_comparison`). Denominators for empty sets, fallbacks and failures are kept fixed; fallback width 2 applies to all arms; `known_zero` and `oracle_f_05` coincide on the two laws with `f* = 0` (implementation check).

---

## 7. Windows cut from one record (article Section 5.1, Supplementary Section S3, Table S1)

Configuration: `extensions/configs/addendum_e7_e9.json`, key `e8`.

Generator. Only mixing stationary processes (periodic mixtures are stationary but not ergodic, so one record does not reveal the population law). Latent symmetric two-state Markov chain with `s` in {0.2, 0.5, 0.8}, detector (0.95, 0.20), `phi` in {0, 0.02}: six cells (`condition_id` 0-5 ordered by `s` then `phi`); one further cell with an asymmetric chain, `m = 0.2`, `rho = 0.6`, `phi = 0.02`. `P(Y_t = 1 | X_t = 1, X_{t-1}) = detect[X_{t-1}]`, `phi` when `X_t = 0`; the chain starts from `X[-1] ~ Bernoulli(m)`.

Samples. Record length 10,002, hence 10,000 consecutive windows (the observation count of the independent baseline); `B = 500` audit centers by simple random sampling without replacement; for `phi > 0`, `K = 500` calibration windows from a separate independent record of the same length; the independent baseline generates 10,000 episodes, 500 audits and 500 calibration episodes separately. Random numbers `SeedSequence([20261014, 11, 3, condition_id, seed, cohort_id])`, cohorts 0 (record), 1 (audit priorities), 2 (calibration record), 3 (calibration priorities), 4-6 (independent baseline), 9 (analysis bootstrap). Each center receives one uniform priority, and the `B = 500` audits are the smallest priorities within the eligible set of each method, so the methods are coupled.

Methods. (1) independent-episode baseline; (2) all overlapping windows treated as independent (`naive_overlap`); (3) thinning to every third center with a fresh selection of 500 audits from the thinned set, calibration thinned the same way; (4) thinning to every tenth center, likewise; (5) and (6) block means over consecutive blocks of 50 or 200 centers: observation moments are the equal-weight means of the block means of each Bernoulli feature; the audits are the same 500 as the naive method, attributed to their blocks, with equal weights over nonempty blocks. Block confidence intervals: for J block means in [0, 1], center `mean(z_j)`, two-sided radius `sqrt(log(2/delta) / (2J))` intersected with [0, 1], `delta` = moment error / 11, J = number of blocks (observation) or nonempty blocks (audit). Calibration for blocks: one-sided radius `sqrt(log(1/0.01) / (2 J_cal))`; other methods use the Clopper-Pearson upper limit. `phi = 0` cells use `epsilon = 0` without calibration. Independence between blocks is an approximation, and no coverage guarantee is claimed for the block means.

Evaluation. Seven cells x 1000 replicates x six methods, seeds 440001-441000; the population target of every method is the N interval of the same law at the true `f` and the same `theta`. Monte Carlo Clopper-Pearson intervals of `contains_population`: for the naive method at error 0.05/6 per cell over the six cells (`naive_overlap_check`: `not_detected` when no upper limit is below 0.95); for thinning to every tenth window and for blocks of 200, at error 0.05/12 over the 12 cell-method combinations (`treatment_check`: `large_loss_excluded` when all lower limits are at least 0.93). Width: the mean width of each treatment minus the independent baseline, with seed-paired 10,000-resample bootstrap 95% intervals per cell. Treatment used for the occupancy record (`selection`): among the treatments with `large_loss_excluded`, the one with the smaller equal-weight mean of fallback-inclusive mean widths over the six cells; this is not a certification of independence or coverage on real data.

---

## 8. Recorded occupancy series (Supplementary Section S13, Table S16)

Configurations: `extensions/configs/addendum_e7_e9.json`, key `e7`; `extensions/configs/room_occupancy_manifest.json`.

Data. UCI Room Occupancy Estimation (CC BY 4.0): a public record with a binary true state recorded by a means other than the detector, a detector present in the raw data, and several days. The nominal 30 s spacing varies (31 s in 63% of consecutive readings, 30 s in 36%, some 61 s), so the raw file is resampled to a 60 s grid (within each local wall-clock minute, occupancy count = maximum, each sensor = maximum, hence OR; minutes without readings are not created, no interpolation). Time zone: the collecting institution's local time.

Definitions. `X_t` = occupied (count above 0); `Y_t` = detector output (OR of the passive-infrared sensor columns); window `(t-1, t, t+1)` with exactly equal spacing, within one day; stratum: local weekdays 09:00 to 17:00 with all three points inside (904 windows); five-minute windows are a separate sensitivity analysis without interpolation. Duplicate time stamps are rejected; rows with missing or invalid values are removed and spacing rechecked.

Evaluation (finite population). The whole stratum record is the auditable population; the record-level `theta` is the mean first-pair spin product over all eligible windows; reference intervals under M, N and F are computed from the empirical window law, with the stationarity residual and the exact program status recorded (an empty or failed reference leaves `contains_population` undefined). Audit draws: `B` in {50, 200, 500} and `K = B`, drawn without replacement and disjoint in each of 1000 replicates, seeds 460001-461000, `SeedSequence([20261014, 11, 5, budget_index, seed, cohort_id])` (the first `B` priorities are the audit, the next `B` the calibration; the five-minute analysis uses design identifier 6); the learned upper limit is always used; if the frame has fewer than `2B` windows the budget is recorded as `insufficient_population` (this applies to `B = 500`, and to every budget after thinning to every tenth window, which leaves 92 windows). The reported analysis therefore uses all 904 windows with the Clopper-Pearson procedure and is descriptive: the record's `f`, the conditional false-negative rate `P(Y = 0 | X = 1)`, `theta`, reference residual and status, raw empty, failure and fallback counts, mean widths and ratios, point-target containment, and interval containment only when the reference is defined. No real-world coverage statement is produced.

Sensitivity to stratum and lag: `extensions/scripts/occupancy_sensitivity.py` (weekdays or all dates; 09:00-17:00 or all hours; lags 1, 2, 3 and 5 minutes; one random stream per replicate from 470001).

---

## 9. SPHERE house recordings (Supplementary Section S13, Table S17)

Implementation: `extensions/scripts/sphere_pir.py`; data: `data/raw/sphere_house/` (see `DATA_SOURCE.md` there).

Data. The SPHERE house recordings (Tonkin et al. 2023; terms of use in `data/raw/sphere_house/DATA_SOURCE.md`) have passive-infrared sensors and room locations annotated from a head-mounted camera, independently of the sensors. The analysis is descriptive: a scripted record of ten adults in one house.

Variables. Only the `raw` view of the dataset is used (19 sessions; the files are stored under `data/raw/sphere_house/raw/NNN/`).
- Room: the living room. The kitchen is a secondary room.
- Grid: cells `[c g, (c+1) g)` from the session start, with `g = 10 s`.
- `X_c = 1` if the mean over the session's annotators of the share of the cell labeled "living" exceeds 0.5. A cell is unlabeled if the mean labeled share (any room) is at most 0.5.
- `Y_c = 1` if a living-room activation interval in `pir_raw.csv` overlaps the cell.
- Windows: `(c-k, c, c+k)` within one session, with `k = 1`. In the primary analysis they are non-overlapping (centers `k, 3k+1, ...`). Windows touching an unlabeled cell are dropped.
- Record target: `theta_hat = 1 - 2 #(X_{c-k} != X_c) / n`.

Procedure.
- Observation moments come from all windows.
- Each of 1000 draws (seed `480001 + r`) takes `B` audit and `B` calibration windows without replacement, disjoint.
- The budget is the one-sided Clopper-Pearson upper limit (0.99) on the calibration windows. The four-state program with simultaneous Clopper-Pearson moment intervals (error 0.04) is primary; the three-state program (budget 0) is a contrast.
- `B` in {50, 100, 200}, evaluated only if `n >= 2B`. An empty set is reported as the fallback `[-1, 1]` and counted; a solver failure stops the run.

Reported: containment of the record's target by the neighbor interval (the draws are independent given the record, so this is design-based containment for this record only); mean widths; shares of draws with upper endpoint below 1 and lower endpoint above 0; `f_hat`, `P(Y=1 | X=0)`, the miss rate `P(Y=0 | X=1)`; the share of window centers at which the sensor of the unused second bedroom is active; the exact reference intervals when feasible, with the stationarity residual.

Sensitivity settings:
1. `(g, k)` in {(5 s, 1), (5 s, 2), (10 s, 3), (20 s, 1), (30 s, 1)};
2. overlapping windows;
3. kitchen;
4. onset-only record (an activation starts in the cell);
5. activations shifted by -2, -1, +1, +2 s;
6. audit and calibration labels from annotator 0 or 1 alone, with the target unchanged;
7. the three-state program.

---

## 10. Recorded data with a reference state (article Section 7, Tables 4 and 5, Figs. 4-7; Supplementary Section S12, Tables S9-S15, Fig. S4)

Implementation: `cohorts/cohort_study.py`; data: `data/raw/recorded_data/DATA_SOURCE.md` (not redistributed); stored results: `manuscript/supplement/cohorts/<name>_cohort.json` for the six data sets `prism`, `prism2`, `border`, `channel_islands`, `caltech` and `offices`.

Data sets.
- PRISM, PRISM2 and Border malaria cohorts: occasion = routine visit; `Y` = parasites by microscopy; `X` = microscopy or loop-mediated isothermal amplification positive in PRISM, quantitative PCR in PRISM2 and Border; consecutive occasions of a window are 21-35 days apart; unit = household.
- Channel Islands and Caltech camera traps: occasion = one trigger sequence of a camera; `X` = a human label of the sequence names an animal; `Y` = the largest animal confidence of the published detector is at least 0.9 (0.80, 0.85 and 0.95 as a sensitivity analysis); unit = camera location.
- Two offices: occasion = five-minute interval; `Y` = motion counter positive; `X` = a person in the labelled image at the start or the end of the interval; unit = one room on one day (room-day).

Windows and target. A window is three consecutive occasions of one series; windows do not overlap. The target is the lag-one correlation of the reference state `X`, pooled over the two adjacent pairs of all windows of a data set; it is held fixed while units are resampled.

Evaluation. For each data set and each number of labels (200, 400, 1000, 2000; 400 in the main tables; a number above 80% of the windows is skipped), each of 1000 replicates draws the units with replacement and reveals the reference state at the central occasion of `labels` randomly chosen windows. Every procedure receives the record of all windows. An empty confidence set counts as a failure of coverage and does not enter the mean width.

Procedures.
- Finite-sample interval: the four-state program with simultaneous Clopper-Pearson moment intervals; half of the labels audit and half calibrate the false-positive budget (error levels 0.04 and 0.01); the reported range of the correlation is an outer range, computed over 48 cells of the prevalence range (largest and smallest target and prevalence term per cell). Computed for M, N and F.
- Plug-in interval: the identified interval of the four-state program at the sample moments, with all labels and the sample false-positive share as the budget.
- Percentile and Imbens-Manski intervals: from the plug-in interval over 200 unit-bootstrap samples that keep the audit status of every window; all labels.
- Four hidden Markov models (one detection probability; detection probability and false-positive rate; two detection probabilities that depend on the previous latent state; a mixture of two chains): fitted by maximum likelihood to the windows, with unit-bootstrap percentile intervals over 200 refits (100 for the two-class mixture, which is started from four random points).
- The record alone (unit-bootstrap percentile interval, 1000 samples).
- Adjacent-pair audit: the reference state of two adjacent occasions is revealed in `labels`/2 windows (unit-bootstrap percentile interval, 2000 samples).

Population intervals. The empirical window law is projected by least squares onto the stationary window laws whose false-positive mass does not exceed that of the data; the M, N and F population intervals are computed at this projection (a total-variation projection is stored alongside). Their percentile ranges use 1000 unit-bootstrap samples.

Two extreme laws. For each data set, the two stationary laws with the smallest and the largest target among those with the full audit law of the projection are computed by the four-state linear program; 1000 samples of independent windows of the size of the data set are drawn from each and evaluated with 400 labels.

Other window definitions and operating points (300 replicates each; 400 labels, or 200 when 400 exceeds 80% of the windows): one window per series or per household, visits 24-32 days apart, age groups, sites, calendar years, other reference definitions, occasions two steps apart, detector thresholds 0.80-0.95, single rooms, hours of the day, another motion-count threshold, and windows without a malaria episode.

Random numbers. `default_rng([940001, r, labels, pool])` for the evaluation replicates of a window pool, `default_rng([950001, r, law])` for the samples from the two extreme laws, `default_rng([930001, r, pool])` for the unit bootstrap of the population quantities, and `default_rng(20261118)` for the random choice of one window per series. One generator per job, so the results do not depend on the number of processes.

Key names in the stored files (`households`, `participants`, `routine_visits`, `one window per participant`) are explained in the README under "Keys inside the result files".

Positive control (Supplementary Section S12). For each data set and each of the three single-chain hidden Markov models, the model is fitted by maximum likelihood to the fully labelled data set; 300 samples of independent windows of the size of the data set are drawn from the fitted model (seed 970001), and all models and the neighbor finite-sample interval are applied with 400 labels. Results: `manuscript/supplement/cohorts/<name>_control.json`; command: `python cohorts/cohort_study.py <name> --positive-control --output DIR --workers W`.
