# Released data

This directory contains the clean tabular data needed to reconstruct the
paper's training samples and condition-stratified analyses.

## Files

- `panel/training_monthly_panel.csv.gz`: 623,740 rows (9,596 fixed sites by
  65 calendar months, January 2021 to May 2026). It contains the ten
  Sentinel-2 history bands and the two quality fields consumed by the sample
  materializer.
- `static/fixed_site_static_metadata.csv`: one row per fixed site, containing
  registered coordinates and the 2021 ESA WorldCover fractions and derived
  classes used for mapping and land-cover stratification.
- `points/point_registry.csv` and `.parquet`: exact fixed-site coordinates for
  replication and Earth Engine upload.
- `temporal_split_v1.csv`: target-month training, validation, and testing
  roles with target-valid sample counts.

## Model-input boundary

The main model uses only the ten historical Sentinel-2 reflectance bands
together with causally constructed history mask, observation age, and calendar
phase. Coordinates and WorldCover fields are not model inputs. Static metadata
is supplied separately only to reproduce maps and condition-stratified
evaluation.

Missing monthly reflectance values are retained as empty CSV cells. The
materializer derives history validity from joint finite, nonzero reflectance,
the valid-acquisition count, and the overlap-difference rule, and then constructs
the value, mask, and causal observation-age arrays.

## Materialization

From the repository root, run:

```bash
python scripts/materialize.py \
  --panel data/panel/training_monthly_panel.csv.gz \
  --split data/temporal_split_v1.csv \
  --output data/store
```

Pandas infers gzip compression from the `.gz` extension; no manual extraction
is required.

## Integrity

- uncompressed panel SHA-256:
  `44C042D0B680DEFE5490B54D6794EA859783EDD3379D19F2A6DC244CCFD97949`
- distributed gzip SHA-256:
  `23E1FFF5D411DCBC79BE8DCBE1C147A9BC3806F3DDD90BD05A9E3A68D117FA33`
- static metadata SHA-256:
  `BFA399364164B5AD8DD3C4504A71B772BEB5FCCAFFF22F53919681CCF366F566`

Sentinel-2 source data remain subject to Copernicus and Google Earth Engine
terms. See `docs/DATA_ACCESS.md` for source identifiers, attribution, boundary
information, and reconstruction options.
