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
data/points/       fixed-site coordinate registry
docs/              data-access notes and paper-to-code map
scripts/data/      Sentinel-2 acquisition and panel-construction scripts
scripts/           training, evaluation, and analysis entry points
src/fpmf/          models, data processing, metrics, and interventions
tests/             model, causality, and release-contract tests
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

The repository includes the exact 9,596-site registry and temporal split
summary. The complete derived monthly reflectance panel and paper-result
outputs are not redistributed. The panel can be reconstructed from public
Sentinel-2 imagery using the scripts in `scripts/data/` and a user-provided
Google Earth Engine project.

See [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) for site upload, Sentinel-2
acquisition, boundary, and source-data information.

## Verification

Verify the model and repository contracts without training:

```bash
pytest -q
python scripts/audit_repository.py
```

## Training and evaluation

After reconstructing and materializing the monthly panel, the main workflow is:

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
