# TDC-CfC

Code, experiment configurations, and fixed-site coordinates for the
paper **“Attributing Next-Month Sentinel-2 Reflectance Predictability Beyond
Climatology and Persistence: Information Sources and TDC-CfC.”**

**Authors:** Xing Cao, Qianjian Xu, and Yuqing Pan.

TDC-CfC jointly predicts Sentinel-2 B2, B3, B4, B8, B11, and B12 at 9,596
fixed sites from the preceding 12 complete calendar months. The forecasting
protocol is strictly past-only, and previous predictions are never reused as
inputs to later forecasts.

## Repository contents

```text
configs/           paper experiment and training configuration
data/panel/        complete 9,596-site by 65-month training panel
data/static/       site coordinates and land-cover stratification metadata
data/points/       fixed-site coordinate registry for Earth Engine upload
data/              temporal split and data documentation
docs/              data-access notes and paper-to-code map
scripts/data/      Sentinel-2 acquisition and panel-construction scripts
scripts/           training, evaluation, and analysis entry points
src/fpmf/          models, data processing, metrics, and interventions
tests/             model, causality, and reproducibility tests
examples/          checkpoint-free synthetic input and reference output
```

## Requirements

The reported experiments used Python 3.12.3, PyTorch 2.8.0 with CUDA 12.8,
and an NVIDIA GeForce RTX 4090. CUDA is recommended for training and
full-panel inference. Pretrained weights are not distributed in this repository.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Exact package versions, model sizes, random seeds, training epochs, and
analysis protocols are recorded in
[`configs/paper_experiments.json`](configs/paper_experiments.json).

## Data

The repository includes the complete model-ready source data used to construct
the paper's strict 12-to-1 samples:

- `data/panel/training_monthly_panel.csv.gz`: 623,740 rows covering 9,596
  fixed sites and 65 calendar months from January 2021 to May 2026;
- `data/static/fixed_site_static_metadata.csv`: coordinates and 2021 ESA
  WorldCover fields used only for mapping and condition-stratified evaluation;
- `data/temporal_split_v1.csv`: target-month training, validation, and test
  assignments and target-valid sample counts;
- `data/points/point_registry.csv`: exact fixed-site registry for replication
  and Earth Engine upload.

The monthly panel contains the ten Sentinel-2 history bands and the two
quality fields required by the released sample materializer. Coordinates and
WorldCover attributes are not model inputs. Paper-result outputs and model
weights are not distributed.

### Study-boundary source

The original 500 m sampling frame used the Huanghua City administrative
boundary (`adcode` **130983**) obtained from the official
[Alibaba Cloud DataV.GeoAtlas area selector](https://datav.aliyun.com/portal/school/atlas/area_selector)
(`areas_v3`; underlying map data from AMap Open Platform; data version updated
May 2021). Readers can download the corresponding provider-hosted GeoJSON
directly from:

<https://geo.datav.aliyun.com/areas_v3/bound/130983.json>

Accessed 20 August 2026. The downloaded file has SHA-256
`F7EFF392ADE7A2B2048C553B3C3ADC6268FFF03C6372EE981C55CE67E6D4D1DD`.
Its geometry was verified to be topologically identical to the boundary used
for the study's Earth Engine asset (identical bounds and zero symmetric
difference). DataV states that this map version is provided for learning and
exchange; the boundary is therefore not redistributed here. For exact
frame reconstruction, preserve the downloaded coordinate sequence without
simplification, buffering, datum conversion, or substitution. See
[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) for the coordinate-system caveat
and upload routes. Exact replication should preferably use the released fixed
sites rather than regenerate them from a boundary.

See [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) for site upload, Sentinel-2
acquisition, boundary, and source-data information.

## Verification

Verify the model and repository contracts without training:

```bash
pytest -q
python scripts/check_repository.py
```

Run an independent CPU forward pass without real data or a checkpoint:

```bash
python scripts/run_synthetic_example.py
```

This example uses artificial standardized values and a deterministic
untrained model state to validate the software path and tensor shapes. See
[`docs/SYNTHETIC_EXAMPLE.md`](docs/SYNTHETIC_EXAMPLE.md).

## Training and evaluation

Materialize the released CSV panel into the immutable 12-to-1 training store:

```bash
python scripts/materialize.py \
  --panel data/panel/training_monthly_panel.csv.gz \
  --split data/temporal_split_v1.csv \
  --output data/store
```

The main workflow is then:

```bash
python scripts/baselines.py --store data/store --split validation --output outputs/baselines

python scripts/train.py \
  --store data/store \
  --model H03_HSR_RD_TIMEMIX_CFC_211K \
  --output runs/H03_seed438344685 \
  --seed 438344685

python scripts/refit.py \
  --store data/store \
  --model H03_HSR_RD_TIMEMIX_CFC_211K \
  --selected-epochs 14 \
  --output runs/H03_refit_seed438344685 \
  --seed 438344685
```

Generic GRU, CfC-mmRNN, and SITS Transformer comparators; HSR/RD factorial
models; representation variants; and independent-band models are available
through the same model factory. Paper analyses are implemented by the
`analyze_*.py` scripts. The mapping from manuscript results to code entry
points is provided in
[`docs/PAPER_CODE_MAP.md`](docs/PAPER_CODE_MAP.md).

After training or refitting, `scripts/infer.py` can run one or more saved
models on a user-supplied NPZ batch that follows the released input contract.

## Citation

If you use this code or fixed-site registry, please cite the
accompanying paper. Complete citation information will be added after
publication.

## License

This repository is released under the MIT License. Sentinel-2 imagery remains
subject to the original Copernicus and Google Earth Engine terms.
