# Paper-to-code map

| Paper component | Implementation | Entry point |
|---|---|---|
| Sentinel-2 acquisition and monthly compositing | `scripts/data/` | `export_fixed_point_multiband_s2_v2_1.py`, `merge_s2_exports.py` |
| Fixed sites, split, and 12-to-1 samples | `data/points/`, `src/fpmf/data.py` | `build_temporal_split.py`, `scripts/materialize.py` |
| PointMean, PointMonth, LastValid, Lag-12, PointMonth+AR(1) | `src/fpmf/classical.py` | `scripts/baselines.py` |
| HGB historical-information model | `src/fpmf/classical.py` | `scripts/train_hgb.py` |
| Plain GRU, generic CfC-mmRNNs, SITS Transformers | `src/fpmf/models/comparators.py` | `scripts/train.py`, `scripts/refit.py` |
| TDC-CfC and HSR/RD factorial cells | `src/fpmf/models/tdc_cfc_base.py`, `tdc_cfc.py` | `scripts/train.py`, `scripts/refit.py` |
| Phase, 6/9-month history, missingness, and target-band variants | `src/fpmf/models/paper_variants.py` | `scripts/train.py`, `scripts/refit.py` |
| TDC-GRU backbone replacement | `paper_variants.py` (`TDC_GRU_212K`) | `scripts/train.py`, `scripts/refit.py` |
| Joint vs. six independent models | `src/fpmf/models/paper_variants.py` | `scripts/train.py`, `scripts/refit.py` |
| Historical, missingness, coverage, and observation-age analyses | `src/fpmf/interventions.py` | `scripts/analyze_interventions.py` |
| Exact RD zero/deviation-sign reversal checks | `src/fpmf/interventions.py` | `scripts/analyze_tdc_components.py` |
| HSR x RD five-seed effects | four trained factorial cells | `scripts/analyze_factorial.py` |
| Spectral heterogeneity, overall/stratified anomaly correlation, PCA | prediction arrays | `scripts/analyze_spectral_structure.py` |
| Final metrics and prediction arrays | `src/fpmf/evaluation.py` | `scripts/evaluate.py` |
| 16-month dependence-aware uncertainty (50,000 paired replicates) | `src/fpmf/bootstrap.py` | `scripts/analyze_bootstrap.py` |
| Leave-one-month-out and support thresholds | prediction arrays | `scripts/analyze_robustness.py` |
| Derived indices, month-band gains, condition strata | prediction arrays and metadata | `scripts/analyze_results.py` |

Representation variants are separate trainable model builders. Post-training
input interventions evaluate how an already trained predictor depends on the
supplied history.

The released contracts preserve paper randomization exactly: cross-model
temporal permutations use seeds 20260727--20260731; final TDC-CfC temporal
shuffling uses fixed order `[7, 0, 10, 3, 5, 1, 11, 8, 2, 9, 4, 6]`;
cross-model nested missingness uses hash seed 20260816; TDC-CfC missingness
stress uses hash seed 20260727; and the hierarchical bootstrap uses seed
20260809 with the formal 16-month support rule. These values are machine-checked
against `configs/paper_experiments.json`.
