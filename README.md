# ECM-402 Lab 4 — CNNs for multi-dimensional signals
GitHub: <add link>  |  Roll no: <fill>

Run: `pip install -r requirements.txt && python src/main.py`
(env vars: `EPOCHS=12`, `TL_EPOCHS=6`). Outputs go to `results/figures` and `results/tables`.

Seed: 42 (splits, stratified 10k/2k subset, weight init, loaders).
Hardware: <CPU-only / GPU model>  |  per-epoch times: see `results/tables/*.csv` (`sec_per_epoch`).
Model selection on validation only; the test set is evaluated once, in Task 5.
Bonus (denoising) not included.
