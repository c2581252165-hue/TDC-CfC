# Synthetic smoke example

The repository includes a small, independently runnable example that requires
neither the Sentinel-2 panel nor a trained checkpoint. It verifies the released
input contract and the TDC-CfC forward path:

```bash
python scripts/run_synthetic_example.py
```

The command loads `examples/synthetic_input.npz`, constructs the reported
211,326-parameter TDC-CfC, fills it with a deterministic **untrained** reference
state, performs a CPU forward pass, and checks the six standardized outputs
against `examples/expected_untrained_predictions.csv`.

The NPZ contains eight artificial samples with:

- 12 historical calendar positions;
- 10 standardized synthetic history channels;
- binary observation masks and causally recomputed observation ages;
- sine/cosine calendar coordinates; and
- no target observations, geographic coordinates, or real Sentinel-2 values.

To save generated predictions without modifying the repository:

```bash
python scripts/run_synthetic_example.py --output /path/to/synthetic_predictions.csv
```

The untrained standardized outputs provide deterministic software-reference
values. Recreate the committed example files with:

```bash
python scripts/create_synthetic_example.py
```
