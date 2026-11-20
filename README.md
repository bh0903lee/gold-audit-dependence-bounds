# Sharp bounds on latent temporal dependence from isolated gold audits: code and data

Code, configurations, stored numerical results and exact certificates for checking the values reported in the article
"Sharp bounds on latent temporal dependence from isolated gold audits: A five-moment identification theorem and a four-point counterexample"
(manuscript submitted to Knowledge-Based Systems). This is release v1.0.

- Code: MIT License (`LICENSE`).
- Numerical outputs: CC BY 4.0, to the extent the authors hold rights in them (`DATA_LICENSE.md`).
- UCI Room Occupancy Estimation data: CC BY 4.0, doi:10.24432/C5P605; see `data/raw/room_occupancy_estimation/DATA_SOURCE.md`.
- SPHERE house data, doi:10.5523/bris.2h0wyctxrd69j2oqccsi45hy1p: not redistributed; `extensions/scripts/fetch_sphere.py` downloads the files the analysis reads from the official repository, whose terms of use apply (`data/raw/sphere_house/DATA_SOURCE.md`).
- Camera-trap, malaria-cohort and office data of Section 9: not redistributed; sources, terms of use and checksums in `data/raw/recorded_data/DATA_SOURCE.md`.

## Requirements

Linux (POSIX: the code uses `resource`, `fcntl` and `ctypes`), Python 3.12, CPU only.

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements-cpu.lock
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
```

All random streams come from NumPy's default generator (PCG64); use the pinned versions to reproduce the numbers exactly.
Only `extensions/scripts/fetch_sphere.py` uses the network, to download the SPHERE files (not redistributed); no other command downloads anything.

## Layout

| Path | Contents |
|---|---|
| `run.py`, `src/topic_experiment/`, `configs/` | main simulation study (word programs), six-variable comparison, assumption-violation study |
| `scripts/` | run plans (`plan_main.py`), parallel execution (`execute_plan.py`), collection of the main-study outputs (`freeze_main_rows.py`), value check of the stored results (`recompute_values.py`), value check (`check_values.sh`), tests (`validate.py`), integrity manifest (`refresh_manifest.py`) |
| `extensions/` | bounded false positives, general stationary laws, overlapping windows, occupancy applications (UCI, SPHERE) |
| `paired_budget/` | paired comparison of the learned false-positive budget with the true false-positive rate |
| `cohorts/` | recorded data with a reference state (Section 9, Supplementary Section S12): camera traps, malaria cohorts, offices; the input files are not redistributed |
| `manuscript/supplement/` | stored numerical results and value files, certificates, certificate verifiers and the analysis scripts of the six-variable comparison and the violation study |
| `docs/STUDY_DESIGN.md` | design of each study: generators, sample sizes, seeds, budgets, methods, reported quantities and random-number keys |
| `data/` | UCI Room Occupancy Estimation data (original archive; the extracted CSV and the 60-second resampled CSV are regenerated from it); sources, checksums and download instructions of the SPHERE house files and of the recorded data of Section 9 (not included) |
| `tests/`, `extensions/tests/`, `paired_budget/tests/`, `cohorts/tests/` | unit tests |

## Quick check (minutes)

Run from the package root.

```sh
python3 -I manuscript/supplement/check_stored_certificates.py   # standard library only
sh scripts/check_values.sh                                       # certificate checks and the full value check in a temporary copy
python scripts/validate.py                                       # all unit tests (one is skipped because the simulation arrays are not included), offline
python run.py --stage verify                                     # file integrity against repository_manifest.json
python run.py --stage smoke --out results/generated/smoke        # 120-row smoke run of the main study
```

## Reported values, inputs and checks

The package is a value-checking package: it contains the stored results, an inventory of every number the article prints from them, the code that checks both, the certificate verifiers, the experiment code and the tests. It does not contain code that typesets the article's tables or draws its figures. `scripts/recompute_values.py` reads only files in `manuscript/supplement/` (and, for their digests, the UCI archive and, if downloaded, the SPHERE files in `data/raw/`), runs no experiments and exits nonzero on any failure of its three steps:

1. Integrity: every file in `manuscript/supplement/` must be listed with its SHA-256 digest in `supplement_manifest.json`, and every listed file must exist (a missing, extra or changed file fails); the UCI archive must match the digest in `extensions/room_occupancy_manifest.json`; the 81 SPHERE files, once downloaded, must match the digests in `extensions/sphere_summary.json` (without them this check is printed as NOT RUN, and the other checks are unaffected).
2. Recomputed value files: the files in the first column below are recomputed from their inputs and compared with the stored ones (byte for byte, or numbers within relative tolerance 1e-9).
3. Reported numbers: `reported_values.json` lists every number printed in the article and its supplementary tables that comes from the stored results (3217 entries: each table cell and each number in the text, including the disjoint Clopper-Pearson pair recorded for each empty confidence set in `empty_set_witnesses.csv`), as printed, with an expression that reads it from the stored files; each value must round to the printed number (or satisfy the stated bound for numbers printed as "below", "at most" or "at least"). Expressions may only index, do arithmetic and call a fixed list of functions.

| Article item | Recomputed value file (in `manuscript/supplement/`) | Inputs (in `manuscript/supplement/`) |
|---|---|---|
| Fig. 2, Table 2, Tables S4 and S5; coverage, containment, slack and fallback numbers of the simulation study | `main_study_summary.json` (widths by budget in key `grid`), `population_summary.json`, `main_study_full_audit_fallbacks.csv` | `main_study_rows.csv`, `main_study_slack_attribution.json`, `main_study_independent_replay.json` |
| The nine empty confidence sets and their disjoint Clopper-Pearson pairs | `main_study_summary.json` (key `empty_sets`) | `main_study_rows.csv`, `empty_set_witnesses.csv` |
| Fig. 3, Supplementary Fig. S1, Tables S2, S3 and S7; numbers of the six-variable comparison and the assumption-violation sections | `sixvar_violation_numbers.json`; table entries are checked through `reported_values.json` | `sixvar_violations/analysis_sixvar.json`, `sixvar_violations/analysis_violations.json` |
| Supplementary Figs. S2 and S3, Tables S1, S6, S8 and S16; numbers on bounded false positives, overlapping windows, the 80 further stationary laws, the learned false-positive budget and the UCI occupancy record (Sections 6, 7.1, 8.2 and 8.6, Supplementary Sections S3, S7, S11 and S13) | `extension_numbers.json`; table entries are checked through `reported_values.json` | `extensions/*_summary.json`, `extensions/fp_gap_certificate.json` |
| Tables 3 and 4, Figs. 4-7, Supplementary Fig. S4, Tables S9-S15; numbers of Section 9 and Supplementary Section S12 (recorded data with a reference state) | (checked through `reported_values.json`) | `cohorts/<name>_cohort.json` for `channel_islands`, `caltech`, `prism`, `border`, `prism2`, `offices`, and `cohorts/<name>_control.json` for the positive control of Supplementary Section S12 (aggregate results only) |
| Table S17 (SPHERE house recordings) | (checked through `reported_values.json`) | `extensions/sphere_summary.json` |
| Paired comparison with the true false-positive rate (Supplementary Section S7: median 3%, 1.1 percentage points, 0.96-1.26, 0.06-3.80) | (checked through `reported_values.json`) | `extensions/paired_budget_summary.json`: `bank_comparison.bank_C_oracle05_median`, `primary.point`, `primary.ci_95`, `law_panel_sensitivity.ci_95` |
| Occupancy sensitivity to stratum and lag (Supplementary Section S13, Table S16) | (checked through `reported_values.json`) | `extensions/application_sensitivity_B200.json`, `extensions/application_sensitivity_B50.json` |
| Eight-moment check on the 512 rational laws (Section 6: 245 strict gaps, largest difference) | (checked through `reported_values.json`) | `extensions/fp_population_rows.jsonl` (the per-law output `population.jsonl` of the `d1` run) |
| SHA-256 digests of the main-study inputs | (no computation) | `main_study_input_manifest.json` (paths refer to the layout of the run archive) |

Fig. 1 is a schematic without data input. The stored inputs are produced from the simulations and the data by the commands under "Re-running the simulations".

## Certificates

| Item | Command | Files |
|---|---|---|
| Four-point separation: four exact rational LP certificates | `python3 -I manuscript/supplement/check_stored_certificates.py` | `verify_theorem.json` |
| Numerical check of Supplementary Section S1: 200 laws, 600 gluings; rebuild of the four certificates | `python manuscript/supplement/verify_theorem.py` | writes `verify_theorem.json` |
| Gap and gluing certificates of the bounded-false-positive program | `python manuscript/supplement/extensions/verify_fp_certificates.py` | reads `extensions/fp_gap_certificate.json`, `extensions/fp_gluing_certificate.json`; writes `extensions/fp_certificate_verification.json` |

The two verifiers rewrite their JSON output next to them; `scripts/check_values.sh` runs them on a temporary copy and compares the output with the stored files.

## Re-running the simulations

Use new output paths under `results/generated/` when working inside the package. The main and extension runners enforce their documented output-root checks. The standalone occupancy-sensitivity, SPHERE and resampling scripts overwrite their named output files if those paths already exist; choose fresh paths for them.

| Study | Commands | CPU time (one thread) |
|---|---|---|
| Main simulation study (200 seeds) | `python scripts/plan_main.py --out results/plans/main.json`; `python scripts/execute_plan.py --plan results/plans/main.json --out results/generated/main --workers 8`; `python scripts/freeze_main_rows.py --runs results/generated/main --out DIR` (writes the row, slack-attribution and input-manifest files to `DIR`; the first two equal the stored files) | about 0.4 h |
| Six-variable comparison (`w3`) / violation study (`w2a`, `w2b`) | `python scripts/plan_main.py --experiment w3 --out results/plans/w3.json` (likewise `w2a`, `w2b`); `python scripts/execute_plan.py --plan results/plans/w3.json --out results/generated/w3 --workers 8`; then `manuscript/supplement/sixvar_violations/analyze_sixvar.py --run results/generated/w3 --plan results/plans/w3.json --out ...` (it writes nothing unless every cell holds all 200 width-comparison seeds) and `analyze_violations.py --run-a ... --run-b ... --out ...` (`reanalysis_main_arrays.py --runs results/generated/main --out ...` applies the six-variable methods to the main-study arrays) | about 1.9 h |
| Bounded false positives (population check `d1`; learned-budget simulation `d2`) | `PYTHONPATH=extensions/src python -m c11_upgrade.cli d1 --full --output DIR` (`DIR/population.jsonl` equals `manuscript/supplement/extensions/fp_population_rows.jsonl`); `... d2 --full --output DIR` | 1 min; 3 min |
| General stationary laws (`e9`) / overlapping windows (`e8`) | `PYTHONPATH=extensions/src python -m c11_upgrade.addendum_cli e9-full --output results/generated/extensions/e9_full`; `... e8-full --output results/generated/extensions/e8_full` | 9 min; 5 min |
| Occupancy application (`e7`; Supplementary Section S13, Table S16) | first regenerate the 60-second file (the three commands in `data/raw/room_occupancy_estimation/DATA_SOURCE.md`, seconds); then `PYTHONPATH=extensions/src python -m c11_upgrade.addendum_cli e7-full --source data/raw/room_occupancy_estimation/derived_60s/occupancy_60s.csv --manifest extensions/configs/room_occupancy_manifest.json --application-only --output results/generated/extensions/application`; the thinning analysis uses `--e8-run results/generated/extensions/e8_full` instead of `--application-only` | about 1 min each |
| Occupancy sensitivity to stratum and lag (Supplementary Section S13, Table S16) | after the 60-second file exists: `python -I extensions/scripts/occupancy_sensitivity.py data/raw/room_occupancy_estimation/derived_60s/occupancy_60s.csv OUT.json 1000 200` and again with `50`; outputs equal `manuscript/supplement/extensions/application_sensitivity_B200.json` and `_B50.json` byte for byte | about 3 min each |
| SPHERE house recordings (Supplementary Section S13, Table S17) | first download the files with `python -I extensions/scripts/fetch_sphere.py`; then `python -I extensions/scripts/sphere_pir.py data/raw/sphere_house/raw OUT_DIR 1000 40`; `OUT_DIR/summary.json` equals `manuscript/supplement/extensions/sphere_summary.json` byte for byte with the pinned versions (other NumPy/SciPy versions differ below 1e-13 relative, which leaves the values in Table S17 unchanged) | about 30 s with 40 processes |
| Recorded data with a reference state (Section 9, Supplementary Section S12) | first obtain the input files as described in `data/raw/recorded_data/DATA_SOURCE.md` (they are not redistributed); then `python cohorts/cohort_study.py <name> --output results/generated/cohorts --workers 38` for each of `channel_islands`, `caltech`, `prism`, `border`, `prism2`, `offices` (`--workers` is the number of parallel processes; the default is 8) The positive control is run with the additional option `--positive-control` (a few minutes per data set). | about 40 min each with 38 processes |
| Paired budget comparison (`p2`) | `PYTHONPATH=paired_budget/src:extensions/src python -m c11_followup.p2_cli p2-full --output results/generated/paired_budget/full --max-cpu-seconds 7200` | about 30 min |

Notes.
- Main-stage runs require `--execute-main` and a plan from `scripts/plan_main.py`; the plan binds the job configuration and every hashed repository file by SHA-256, and `run.py` re-checks it before running. Write plans under `results/` (not hashed; `plan_main.py` refuses other locations inside the package) or outside the package.
- Empty confidence sets and solver failures are kept apart. A linear program that the solver certifies as infeasible is an empty confidence set and is recorded as the flagged conservative interval [-1, 1] (`empty_confidence_set`, `error`). In the main simulation runner any other unsuccessful solver status (iteration limit, numerical difficulty) raises `SolverFailure`: the job fails, its manifest records the status and message, and nothing is counted as an interval or an empty set; the SPHERE script also stops on a solver failure. The extension, paired-comparison and occupancy-sensitivity summaries count solver failures separately from empty sets (`solver_failure_n`, `solver_failures`) and never count them as covering. In the runs reported in the article every empty set was a certified infeasibility and no solver failure occurred.
- Output roots of the extension and paired-comparison runners: environment variables `EXTENSIONS_OUTPUT_ROOT` and `PAIRED_BUDGET_OUTPUT_ROOT` (defaults `results/generated/extensions` and `results/generated/paired_budget`). Outputs must be new subdirectories of these roots.
- The extension runners check SHA-256 bindings of their source files (`extensions/configs/addendum_e7_e9.json`, `paired_budget/configs/p2.json`). After editing any file, run `python scripts/refresh_manifest.py`; after editing a bound source file, also update the corresponding hash.
- Seeds: `configs/*.json` (main study 91001-91200; six-variable comparison and violation study 92001-93200; smoke and poc seeds are disjoint), `extensions/configs/*.json` and the extension modules, `paired_budget/configs/p2.json` (480001-480200); occupancy sensitivity to stratum and lag, one stream per replicate from 470001; SPHERE recordings, one stream per draw from 480001 (a different generator construction from the paired comparison, so the streams are unrelated); recorded-data study (`cohorts/cohort_study.py`), one stream per replicate from `[930001, r, pool]` (unit bootstrap of the population quantities), `[940001, r, labels, pool]` (evaluation) and `[950001, r, law]` (samples from the two extreme laws), and seed 20261118 for the random choice of one window per series.
- The extracted occupancy CSV and the 60-second file are not stored; `data/raw/room_occupancy_estimation/DATA_SOURCE.md` gives the three commands that regenerate them from the included archive, with the expected SHA-256 values.

## Names in commands and result files

Each study is selected by a short name in its command or configuration, and the same name labels its keys in the result files. The commands below select the studies described in the article. Stored result keys are retained for compatibility with the checking scripts.

| Study | Command or configuration | Name in result files |
|---|---|---|
| Main simulation study | `configs/main.json`, `scripts/plan_main.py` (no `--experiment`) | `topic` "C11" and job IDs `C11-seed-<seed>` (all configurations); `stage` "v42" (`main_study_rows.csv`), `collection` "v4_2" (`main_study_summary.json`), run-archive paths `results/runs/v4_2/...` (`main_study_input_manifest.json`); dataset "reanalysis_v42" = six-variable methods applied to the main-study arrays |
| Six-variable comparison | `--experiment w3`, `configs/main_w3.json` | `w3` |
| Assumption-violation study (1000-seed set, 200-seed set; analyzed separately) | `--experiment w2a`, `--experiment w2b` | `w2a`, `w2b` |
| Bounded false positives: population check, learned-budget simulation | `c11_upgrade.cli d1`, `d2` | `d1`, `d2` |
| General stationary laws, overlapping windows, occupancy application | `c11_upgrade.addendum_cli e9-full`, `e8-full`, `e7-full` (configuration `extensions/configs/addendum_e7_e9.json`) | `e9`, `e8`, `e7`; `handoff` = occupancy application with the method selected by the overlapping-window study |
| Paired budget comparison | `c11_followup.p2_cli p2-full`, `paired_budget/configs/p2.json` | `p2` |

`c11_upgrade` and `c11_followup` are the Python packages in `extensions/src/` and `paired_budget/src/`; `src/topic_experiment/w3w2.py` runs the six-variable comparison and the violation study.

Keys inside the result files:

- `run` fields (`analysis_sixvar.json`, `analysis_violations.json`, `sixvar_violation_numbers.json`): run directories, relative to the package root, from which the stored analyses were computed; a re-analysis reads the directories passed to it.
- `width_comparison`, `extended`, `reanalysis_v42` (six-variable comparison): the 200 width-comparison seeds, these plus 800 further seeds, and the six-variable methods applied to the main-study arrays.
- `coverage_check`, `naive_overlap_check`, `treatment_check`, `containment_lower_below_093`, `narrower_laws`: Monte Carlo summaries of containment and width defined in `docs/STUDY_DESIGN.md`.
- V1, V2a, V2b, V3a, V3b, V4a, V4b, V5, V6: violation axes (docstring of `src/topic_experiment/violations.py`).
- bank A / bank C: three-state laws without false positives / four-state laws with a learned false-positive budget.
- M / N / F: marginal / neighbor / full audit information; N6, N8, M8: information sets (m, a, b, j), the eight central moments, pair-LP moments.
- `theorem_status` in `fp_population_summary.json` is a fixed label written by the `d1` runner; the eight-moment result is Proposition 5 of the article.
- `cohorts/<name>_cohort.json` (Section 9): the six data sets share one result format and one set of key names, taken from the malaria cohorts. `households` = sampling units (households; camera locations; one room on one day); `participants` = series (participants; camera locations; room-days); `routine_visits` = occasions (routine visits; trigger sequences; five-minute intervals); the pool `one window per participant` = one window per series; the top-level key `cohort` and the file names `*_cohort.json` name the data set, also for the camera traps and the offices. `median_days_between_visits` and `share_of_gaps_21_to_35_days` are meaningful for the malaria cohorts only (for the other data sets the occasions are numbered consecutively, which gives the values 1 and 0).

## Data

The original UCI Room Occupancy Estimation archive is included; its extracted CSV and the 60-second resampled file are regenerated by three commands (citation, DOI, license, checksums, commands and the resampling rule in `data/raw/room_occupancy_estimation/DATA_SOURCE.md`). Raw simulation arrays and per-replicate files are not included because of their size (about 0.7 GB); the commands above regenerate them exactly from the configurations and seeds. The SPHERE house recordings (doi:10.5523/bris.2h0wyctxrd69j2oqccsi45hy1p) are not redistributed: `python -I extensions/scripts/fetch_sphere.py` downloads the 81 files the analysis reads from the University of Bristol data repository and checks them against the stored SHA-256 digests; `data/raw/sphere_house/DATA_SOURCE.md` gives the citation, terms of use, download URLs and checksums. The stored SPHERE results and the values of Table S17 are checked without these files. The recorded data of Section 9 (LILA BC camera-trap label and detector files, ClinEpiDB malaria cohorts, Zenodo office data) are not redistributed; `data/raw/recorded_data/DATA_SOURCE.md` lists the sources, terms of use and SHA-256 digests of the files read, and `manuscript/supplement/cohorts/` holds aggregate results only. These raw recorded data must never be added to the package or to an archive of it; `.gitignore` excludes everything under `data/raw/recorded_data/` except `DATA_SOURCE.md`, and the environment variable `RECORDED_DATA` lets the files be kept in a folder outside the package.

## Citation

Please cite the article and this repository:

- B. Lee and K. Kim. Sharp bounds on latent temporal dependence from isolated gold audits: A five-moment identification theorem and a four-point counterexample. Manuscript submitted to Knowledge-Based Systems, 2026.
- B. Lee and K. Kim. gold-audit-dependence-bounds: code and data, release v1.0. https://github.com/bh0903lee/gold-audit-dependence-bounds

For the camera-trap, malaria-cohort and office data of Section 9, cite the sources below and follow the terms summarized in `data/raw/recorded_data/DATA_SOURCE.md`; the ClinEpiDB studies require the acknowledgement given there.

- Channel Islands camera traps: The Nature Conservancy (2021). Channel Islands camera traps 1.0 [Dataset]. https://lila.science/datasets/channel-islands-camera-traps/
- Caltech camera traps: Beery, S., Van Horn, G. & Perona, P. (2018). Recognition in terra incognita. Computer Vision - ECCV 2018, Lecture Notes in Computer Science, Springer, 472-489. https://doi.org/10.1007/978-3-030-01270-0_28
- Camera-trap detector: Beery, S., Morris, D. & Yang, S. (2019). Efficient pipeline for camera trap image review. arXiv:1907.06772.
- PRISM cohort: Kamya, M. R. et al. (2015). Malaria transmission, infection, and disease at three sites with varied transmission intensity in Uganda: implications for malaria control. The American Journal of Tropical Medicine and Hygiene 92(5), 903-912. https://doi.org/10.4269/ajtmh.14-0312
- PRISM2 cohort: Nankabirwa, J. I. et al. (2020). Malaria transmission, infection, and disease following sustained indoor residual spraying of insecticide in Tororo, Uganda. The American Journal of Tropical Medicine and Hygiene 103(4), 1525-1533. https://doi.org/10.4269/ajtmh.20-0250
- PRISM2 Border cohort: Nankabirwa, J. I. et al. (2022). Measures of malaria transmission, infection, and disease in an area bordering two districts with and without sustained indoor residual spraying of insecticide in Uganda. PLOS ONE 17(12), e0279464. https://doi.org/10.1371/journal.pone.0279464
- ClinEpiDB: Ruhamyankaka, E. et al. (2020). ClinEpiDB: an open-access clinical epidemiology database resource encouraging online exploration of complex studies. Gates Open Research 3, 1661. https://doi.org/10.12688/gatesopenres.13087.2
- Offices: Siegl, C., Hirsch, T., Kohl, T., Wotawa, F. & Schweiger, G. (2024). Data and sourcecode from: Neural network-based occupancy detection on the edge. Zenodo. https://doi.org/10.5281/zenodo.11573783

For the occupancy data, cite Singh, A. & Chaudhari, S. (2018). Room Occupancy Estimation [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5P605.
For the SPHERE house data, cite Tonkin, E. L. et al. (2023). A multi-sensor dataset with annotated activities of daily living recorded in a residential setting. Scientific Data 10, 162. https://doi.org/10.1038/s41597-023-02017-1, and the dataset, doi:10.5523/bris.2h0wyctxrd69j2oqccsi45hy1p.
