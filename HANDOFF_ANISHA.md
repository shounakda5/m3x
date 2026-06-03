# Handoff — Anisha (SVM & Random Forest)

Hey Anisha — sorry for the delay on getting the refactored code to you. To make up for it I've run the data pipeline and a first training pass myself so you're not starting from scratch. Here's everything you need to know.

---

## What's already done

### 1. The labeled dataset is ready

**`data/features/features_labeled_new.csv`** — this is your training input. It combines both recording sessions (June 1 and June 2) and is already cleaned and labeled.

| | |
|---|---|
| Total rows | 1,288 (each row is one ~5 s sub-window) |
| Features | 38 physiological features |
| Classes | 4 (all present — eustress is no longer missing) |
| Unknown rows | 8 dropped (blocks recorded after the last survey ended) |

Class breakdown:

| Label | Count | % |
|---|---|---|
| relax | 468 | 36% |
| eustress+distress | 358 | 28% |
| distress | 286 | 22% |
| eustress | 176 | 14% |

The grouping column for cross-validation is `Block_ID`. All sub-windows within the same block always stay in the same fold, so there's no data leakage.

### 2. A baseline training run is done

**`results/baseline/`** — SVM and RF have both been trained and evaluated with 5-fold StratifiedGroupKFold. Results:

| Model | Accuracy | F1 Macro |
|---|---|---|
| Random Forest | **74.5%** ± 2.8% | **0.70** ± 0.03 |
| SVM | 59.1% ± 2.7% | 0.51 ± 0.02 |

RF is already performing well. SVM needs tuning — its hyperparameters in `config.py` were tuned on the old 3-class dataset, so expect it to improve meaningfully once re-tuned on this data.

Per-class F1 for RF (where it matters most):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| relax | 0.76 | 0.85 | 0.80 |
| eustress+distress | 0.76 | 0.81 | 0.78 |
| distress | 0.74 | 0.66 | 0.70 |
| **eustress** | **0.66** | **0.45** | **0.53** |

Eustress is the hardest class (smallest, and physiologically close to eustress+distress). Keep an eye on its F1 — if it stays below 0.50 after tuning, flag it.

Serialized model files from the baseline are in `results/baseline/`:
- `randomforest_model.pkl`
- `svm_model.pkl` + `svm_scaler.pkl`

### 3. A note on the window size sweep

The original plan called for testing five window sizes (3 s, 5 s, 10 s, 30 s, 60 s). That sweep is **not pre-built** into the pipeline and the baseline above was trained at the ~5 s subwindow level — that's the native resolution of Farhad's feature extraction.

To run the sweep properly there are two options, neither of which is fully set up yet:

- **Larger windows (10 s, 30 s, 60 s):** achievable from the Python side by averaging consecutive subwindow rows in the feature CSV. This requires aggregation logic to be added to the preprocessing step, which hasn't been written yet.
- **Smaller or different sizes (3 s, or anything other than ~5 s):** not possible without Farhad re-running the MATLAB feature extraction at a different subwindow size.

For now, focus your tuning effort on the ~5 s data that's already ready. If you want to pursue the sweep, flag it and we can figure out which approach makes sense given the time.

---

## Remaining tasks

### Task 1 — Tune SVM

The SVM baseline is weak (51% F1 macro). This is expected — the params in `config.py` were tuned on old data. Re-tune it:

```bash
python tune.py \
  --data data/features/features_labeled_new.csv \
  --models svm \
  --n-iter 40
```

After tuning, update `SVM_PARAMS` in `config.py` with the new best params from `results/best_params.txt`, then re-run training to get the proper CV scores:

```bash
python train.py \
  --data data/features/features_labeled_new.csv \
  --models svm \
  --output_dir results/svm_tuned
```

### Task 2 — Check RF, tune if needed

RF is already at 0.70 F1 macro, which is solid. Run training independently to confirm:

```bash
python train.py \
  --data data/features/features_labeled_new.csv \
  --models rf \
  --output_dir results/rf_final
```

If F1 macro comes in below 0.65, tune it:

```bash
python tune.py \
  --data data/features/features_labeled_new.csv \
  --models rf \
  --n-iter 40
```

### Task 3 — Identify your best configuration for each model

For each model, you're looking for the (hyperparameters) combination that gives the highest **F1 macro**. Accuracy is secondary. If eustress F1 drops below 0.50, note it even if overall F1 is fine.

Once you have your best config for both SVM and RF, re-run training one final time to produce clean results and serialized files.

### Task 4 — Hand off to Shounak

When you're done, let Shounak know the location of the final:
- `results/<your_dir>/svm_model.pkl` + `svm_scaler.pkl`
- `results/<your_dir>/randomforest_model.pkl`
- `results/<your_dir>/fold_results.csv`

He'll compare these against CNN and XGBoost results and select the overall winning model for deployment.

---

## Useful reference

**Run training (general form):**
```bash
python train.py \
  --data data/features/features_labeled_new.csv \
  --models svm rf \
  --output_dir results/<your_folder_name>
```

**What each output directory contains after training:**
```
results/<folder>/
  fold_results.csv              ← per-fold accuracy and F1
  svm_model.pkl                 ← trained SVM (ready for .tflite export)
  svm_scaler.pkl                ← the StandardScaler fitted on training data
  randomforest_model.pkl        ← trained RF
  RandomForest_feature_importances.csv
```

**Quick sanity check before your first run:**
```python
import pandas as pd
df = pd.read_csv("data/features/features_labeled_new.csv")
print(df["Label"].value_counts())        # all 4 classes should show up
print(df.shape)                          # should be (1288, 49)
```

---

## Notes on data

- Each row in the CSV is one ~5 s sub-window. Two sub-windows come from each 10 s recording block (`SubWindow_ID` = 1 or 2 within the same `Block_ID`).
- `Session_ID` = 1 (June 1 data) or 2 (June 2 data). `Participant_ID` = 501 throughout.
- The `Block_ID` column is globally unique across both sessions (session 2 block IDs are offset so they don't collide with session 1).
- The raw EMA survey files and original feature Excel files are in `data/survey/` and `data/features/` if you need to check anything.

---

*Let me know if anything looks off or if you hit any errors.*
