# Recorded data of Section 7: sources, terms of use and checksums

This package redistributes none of the files below. It contains the analysis code (`cohorts/cohort_study.py`) and aggregate results (`manuscript/supplement/cohorts/`), which hold no participant-, camera- or room-level record. Obtain the files from their providers and place them in `data/raw/recorded_data/` as listed, or set the environment variable `RECORDED_DATA` to another folder with this layout. The files must never be added to the package or to an archive of it (the package's `.gitignore` excludes everything in this folder except this note).

## Malaria cohorts (ClinEpiDB)

Three cohort studies of the Program for Resistance, Immunology, Surveillance and Modeling of Malaria in Uganda, distributed by ClinEpiDB (https://clinepidb.org), access level "Public":

| Cohort | ClinEpiDB study identifier | Release used | Period and site |
|---|---|---|---|
| PRISM ICEMR Cohort | DS_0ad509829e | release 24 | 2011 to 2017, Walukuba, Kihihi and Nagongera sub-counties |
| PRISM2 ICEMR Cohort | DS_51b40fe2e2 | release 28 | October 2017 to October 2019, Nagongera sub-county, Tororo District |
| PRISM2 ICEMR Border Cohort | DS_17191d35b9 | release 33 | August 2020 to September 2023, Tororo and Busia Districts |

Terms of use: the ClinEpiDB Data Access and Use Policy. Among other conditions, the data are for research and statistical analysis only, no attempt may be made to identify participants, the data may not be redistributed without approval, and publications acknowledge the investigators, the funders, the study, its identifier and release, and ClinEpiDB. Download the files from the "Download" tab of each study.

| File | From | Bytes | SHA-256 |
|---|---|---|---|
| `prism/PRISM_cohort_Samples.txt` | `rls0024_PRISM_cohort_Samples.txt.zip` | 3,857,446 | b3035ef3acdc6202fb4d8714f5bd57daa4a44edef3df1233bebfb5c786a60d44 |
| `prism/PRISM_cohort_Participant_repeated_measures.txt` | `rls0024_PRISM_cohort_Participant_repeated_measures.txt.zip` | 13,068,676 | 7294f77572376832274edfbddf2d95e80cec1eb32ea41a39b3532d7a4c9b2b04 |
| `prism/PRISM_cohort_Households.txt` | `rls0024_PRISM_cohort_Households.txt.zip` | 91,443 | aa3a525c46c616a20d1dee3d16d67d3246acefe1209d950f425384d11ceb8dd4 |
| `prism2/PRISM2_cohort_Samples.txt` | `rls0028_PRISM2_cohort_Samples.txt.zip` | 2,642,731 | 0fd4d9ee1d2f994301d14900c9efdd633d800e264201635969ceff77329cad54 |
| `prism2/PRISM2_cohort_Participant_repeated_measures.txt` | `rls0028_PRISM2_cohort_Participant_repeated_measures.txt.zip` | 8,932,177 | a26cd8b92d443519c6fccadb729421eb92f347fee09e4c441aca1fb4a83cf3ba |
| `border/PRISM2_Border_Cohort_All_Visits_database.csv` | `PRISM2_Border_Cohort_All_Visits_database.csv` | 15,730,110 | 6798cdc64150fffe36c857c9916e079085faddd866404b4a46a4a4fd3b9c14e4 |

Variables read: for PRISM and PRISM2, observation date and type, age, asexual parasites by microscopy, parasites by LAMP (PRISM) or by qPCR and its density (PRISM2), participant and household identifiers, and the sub-county of the household (PRISM); for the Border Cohort, `cohortid`, `hhid`, `District`, `date_obf`, `visittype`, `ageyrs`, `BSdich`, `qPCRdich`, `qpcr`, `malariacat`. Dates are shifted by at most seven days per participant by the data providers.

Acknowledgement: we thank the PRISM, PRISM2 and PRISM2 Border Cohort investigators and study teams, the Infectious Diseases Research Collaboration, the district administrations, and the study participants and their families. The studies were funded by the US National Institutes of Health as part of the International Centers of Excellence for Malaria Research program (U19AI089674).

## Camera traps (LILA BC)

Label files and published MegaDetector v5a result files of two collections hosted by LILA BC (https://lila.science), released under the Community Data License Agreement, permissive variant. No image is read. Sources: The Nature Conservancy (2021), Channel Islands camera traps 1.0; Beery, Van Horn & Perona (2018), Recognition in terra incognita, ECCV 2018, doi:10.1007/978-3-030-01270-0_28 (Caltech camera traps); Beery, Morris & Yang (2019), Efficient pipeline for camera trap image review, arXiv:1907.06772 (detector).

| File in `camera_traps/` | From | Bytes | SHA-256 |
|---|---|---|---|
| `channel-islands-camera-traps.json.zip` | https://lila.science/datasets/channel-islands-camera-traps/ | 18,779,444 | 9bdf2bfe330ad5917cf3abaa46d3f03dcebc1c5f959033631c4fd57a5be88cc4 |
| `channel-islands-camera-traps_mdv5a.0.0_results.json.zip` | https://lila.science/megadetector-results-for-camera-trap-datasets/ | 3,487,781 | 83c4362665e9aa1124841dd45016078746f54b6f8cd7cec7081e4f5d1c5c4dda |
| `caltech_camera_traps.json.zip` | https://lila.science/datasets/caltech-camera-traps | 9,013,179 | 976bd9bc9380df7a03018ac70b5340bd2ff81d79a99271058f026a539450d1d6 |
| `caltech-camera-traps_mdv5a.0.0_results.json.zip` | https://lila.science/megadetector-results-for-camera-trap-datasets/ | 5,174,891 | 6caba6a78795a24320e83697fab40cd7dd527a245ae1810611ca704696cf20b6 |

## Two offices (Zenodo)

Source: Siegl, C., Hirsch, T., Kohl, T., Wotawa, F. & Schweiger, G. (2024). Data and sourcecode from: Neural network-based occupancy detection on the edge. Zenodo. https://doi.org/10.5281/zenodo.11573783

| File in `graz_offices/` | From | Bytes | SHA-256 |
|---|---|---|---|
| `occupancy-detection-dataset.zip` | https://doi.org/10.5281/zenodo.11573783 (license AGPL-3.0-or-later) | 787,327 | 3bb3e7c81efcf3d81988933577ddc9e4bdc1289eeff11424747801c0c993b47c |

## Reproduction

With the files in place, the stored results are reproduced by (about 40 minutes each with 38 processes; `--workers` is the number of parallel processes and its default is 8):

    python cohorts/cohort_study.py <name> --output results/generated/cohorts --workers 38

for `<name>` in `prism`, `prism2`, `border`, `channel_islands`, `caltech`, `offices`.
