-GitHub: <git remote add origin https://github.com/rajgigyashu-2412/ecm402-lab4-123110.git>  |  Roll no: <123110>
+GitHub: https://github.com/<github.com/rajgigyashu-2412>/ecm402-lab4-<123110>  |  Roll no: <123110>
 
 Run: `pip install -r requirements.txt && python src/main.py`
 (env vars: `EPOCHS=12`, `TL_EPOCHS=6`). Outputs go to `results/figures` and `results/tables`.
 
 Seed: 42 (splits, stratified 10k/2k subset, weight init, loaders).
-Hardware: <GPU T4>  |  per-epoch times: see `results/tables/*.csv` (`sec_per_epoch`).
+Hardware: Google Colab, <GPU model from run_info.json, e.g. NVIDIA T4>.
+
+Per-epoch wall-clock time (from `results/tables`):
+
+| Model | Sec / epoch |
+|---|---|
+| MLP baseline | 11.9 |
+| Custom CNN (2 blocks, max-pool) | 13.3 |
+| ResNet-18, feature extraction | 7.3 |
+| ResNet-18, layer4 unfrozen | 8.0 |
+| ResNet-18, differential LR | 9.4 |
+
 Model selection on validation only; the test set is evaluated once, in Task 5.
 Bonus (denoising) not included.
