# Data

The simulation studies are synthetic: each episode is generated from the stated latent/detection model, and a gold audit reveals only its central latent state. No input data are needed for them.

The recorded-occupancy application uses the public UCI Room Occupancy Estimation dataset (CC BY 4.0, doi:10.24432/C5P605). The original archive is in `raw/room_occupancy_estimation/`; the extracted CSV and the 60-second resampled file are regenerated from it by the three commands in `DATA_SOURCE.md` there, which also gives the citation, download URL, SHA-256 checksums and the resampling rule.

The SPHERE house recordings (CC BY 4.0, doi:10.5523/bris.2h0wyctxrd69j2oqccsi45hy1p) are in `raw/sphere_house/`: unmodified copies of the 81 files the analysis reads (session metadata, PIR activation intervals and per-annotator room-location intervals), with citation, download URLs and SHA-256 checksums in `DATA_SOURCE.md` there.

Raw simulation arrays and per-replicate files are not included because of their size (about 0.7 GB); they are regenerated exactly by the commands in the top-level README from the configurations and seeds in this package.
