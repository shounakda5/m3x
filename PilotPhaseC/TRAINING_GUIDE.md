# Pilot Phase C — training guide for teammates

This guide explains how to **add a new person's data**, train **four personalized models** per person, train **four generalized models** across everyone (when you have multiple people registered), and **where to find outputs**.

You do not need to understand the ML code — only copy the folder layout, edit one config block, and run the commands below.

---

## 1. One-time setup (per machine)

1. Open a terminal.
2. Go to the **`m3x`** folder in this repo (the parent of `PilotPhaseC/`).
3. Use the project virtualenv Python:

```bash
cd /path/to/m3x
.venv/bin/python -m PilotPhaseC.src.train_classifier --help
```

If that prints help text, the environment is ready. If Python or TensorFlow is missing, ask the team lead to run the `m3x` environment setup once on your machine.

**Always run training from the `m3x` directory** using `.venv/bin/python -m PilotPhaseC...` as shown below.

---

## 2. Where to put new data

### Folder location

Put each new person's raw export here:

```
m3x/PilotPhaseC/data/<YourFolderName>/
```

Example (existing person):

```
m3x/PilotPhaseC/data/Ring_01_PID_104/
```

### Required subfolders (same names every time)

Inside that folder you need **exactly two subfolders** (same Excel format as Ring_01):

| Subfolder name on disk | What it is |
|------------------------|------------|
| `1_All_continous` | All windows labeled across the continuous recording (note spelling: **continous**) |
| `2_labeled_last5min` | Only windows in the ~5 minutes before each survey |

Each subfolder contains one or more **`.xlsx`** files. Do not rename the subfolders unless you also update `config.py` (step 3).

### Excel format (must match Ring_01)

Each row is one sensor sample; **112 rows = one window**. Required columns include:

- `Global_Window`, `Row_In_Window`, `Survey_ID`
- `Label` (classification), `Stress_0to100` (regression)
- `LED1`, `LED2`, `LED3`, `TempC_x100`, `HumidityRH_x100`, `AccX_mg`, `AccY_mg`, `AccZ_mg`

See `README.md` in this folder for the full column table.

---

## 3. Register the new person in `config.py`

Open:

```
m3x/PilotPhaseC/config.py
```

Find the `PARTICIPANTS = { ... }` block. **Copy the existing entry** and add a new one:

```python
PARTICIPANTS = {
    "ring01_pid104": {
        "dir": DATA_DIR / "Ring_01_PID_104",
        "lineages": {
            "all_continuous":    "1_All_continous",
            "labeled_last5min":  "2_labeled_last5min",
        },
    },
    "ring02_pid205": {   # <-- NEW: pick a short id (lowercase, no spaces)
        "dir": DATA_DIR / "Ring_02_PID_205",   # <-- must match your folder name under data/
        "lineages": {
            "all_continuous":    "1_All_continous",
            "labeled_last5min":  "2_labeled_last5min",
        },
    },
}
```

Rules:

- **`ring02_pid205`** — internal key used in commands (`--participant ring02_pid205`). Use something memorable and unique.
- **`dir`** — must point to the folder you created under `PilotPhaseC/data/`.
- **`lineages`** — folder names under that person's directory; keep as above unless data team uses different names.

Save the file. **Every person you want in generalized training must appear in this dict.**

---

## 4. What gets trained (four models per person)

For **each person** and **each lineage** you train:

| # | Model | Script | `--lineage` |
|---|--------|--------|-------------|
| 1 | Classifier (4 mental-state labels) | `train_classifier` | `all_continuous` |
| 2 | Classifier | `train_classifier` | `labeled_last5min` |
| 3 | Regressor (`Stress_0to100`) | `train_regressor` | `all_continuous` |
| 4 | Regressor | `train_regressor` | `labeled_last5min` |

That is **four personalized models** per person.

---

## 5. Commands — personalized models (one person)

Replace `ring02_pid205` with your participant key from `config.py`.

From the **`m3x`** directory:

```bash
PY=.venv/bin/python
PID=ring02_pid205

# 1–2: classification
$PY -m PilotPhaseC.src.train_classifier --participant $PID --lineage all_continuous
$PY -m PilotPhaseC.src.train_classifier --participant $PID --lineage labeled_last5min

# 3–4: regression
$PY -m PilotPhaseC.src.train_regressor --participant $PID --lineage all_continuous
$PY -m PilotPhaseC.src.train_regressor --participant $PID --lineage labeled_last5min
```

### First time or after replacing Excel files

Add **`--force-reload`** so cached windows are rebuilt from Excel:

```bash
$PY -m PilotPhaseC.src.train_classifier --participant $PID --lineage all_continuous --force-reload
```

### How long it takes

Roughly **several minutes per command** (tuning + cross-validation + export). All four runs for one person are often **~30–60 minutes** total depending on data size and machine.

---

## 6. Commands — generalized models (all registered people)

Use this when **two or more** entries exist in `PARTICIPANTS`. Data from every registered folder is combined; `Survey_ID` is prefixed per person so cross-validation does not leak across people.

Same four model types, but **four commands total** (not per person):

```bash
PY=.venv/bin/python

$PY -m PilotPhaseC.src.train_classifier --generalized --lineage all_continuous
$PY -m PilotPhaseC.src.train_classifier --generalized --lineage labeled_last5min
$PY -m PilotPhaseC.src.train_regressor --generalized --lineage all_continuous
$PY -m PilotPhaseC.src.train_regressor --generalized --lineage labeled_last5min
```

Outputs go under `results/generalized/` (see below).

---

## 7. Where trained artifacts and reports live

### Personalized (one person)

```
m3x/PilotPhaseC/results/<participant_id>/<lineage>/classification/
m3x/PilotPhaseC/results/<participant_id>/<lineage>/regression/
```

Example:

```
m3x/PilotPhaseC/results/ring02_pid205/all_continuous/classification/
```

### Generalized (pooled)

```
m3x/PilotPhaseC/results/generalized/<lineage>/classification/
m3x/PilotPhaseC/results/generalized/<lineage>/regression/
```

### Files in each output folder (what firmware / analysis needs)

| File | Purpose |
|------|---------|
| `cnn_1d_model.tflite` | Float32 model for device |
| `cnn_1d_model_int8.tflite` | INT8 quantized model |
| `cnn_1d_scaler.npz` | Input scaling; **classifier also stores `logit_biases`** for final label decision |
| `cnn_1d_model.pkl` | Keras weights (Python reference) |
| `manifest.json` | Run metadata + **pooled OOF metrics** (main numbers for comparisons) |
| `cv_report.txt` | Full per-fold reports, confusion matrix, per-class precision/recall |
| `fold_results.csv` | One row per CV fold |
| `int8_quantization_report.txt` | Quantization scales / arena size estimates |

**Classifier deployment:** after INT8 softmax output, apply the same logit bias step documented in `manifest.json` → `training.logit_biases` (order matches `CLASSES` in `config.py`).

---

## 8. Results summary document (table of all runs)

After training (any mix of people / generalized), regenerate the summary from the repo root **`m3x`**:

```bash
.venv/bin/python -m PilotPhaseC.src.summarize_results
```

This writes:

```
m3x/PilotPhaseC/results/RESULTS_SUMMARY.md
```

Open that file for a **single table** of pooled OOF metrics for every `manifest.json` found under `results/`. For detail, open the matching `cv_report.txt` in the same run folder.

---

## 9. Quick checklist for a new person

1. [ ] Copy folder layout: `data/<Ring_XX_PID_YYY>/1_All_continous/*.xlsx` and `2_labeled_last5min/*.xlsx`
2. [ ] Add entry to `PARTICIPANTS` in `config.py`
3. [ ] Run four personalized commands (section 5) with `--force-reload` the first time
4. [ ] Check `results/<your_pid>/.../manifest.json` and `cv_report.txt`
5. [ ] Run `summarize_results` to refresh `RESULTS_SUMMARY.md`
6. [ ] When multiple people exist, run four **generalized** commands (section 6) and summarize again

---

## 10. If something fails

| Symptom | What to try |
|---------|-------------|
| `No .xlsx files found` | Check folder names (`1_All_continous`, `2_labeled_last5min`) and `config.py` `lineages` paths |
| `Unknown participant` | `--participant` must match a key in `PARTICIPANTS` exactly |
| `missing required column` | Excel columns don't match Ring_01 — compare headers |
| Stale/wrong window counts after editing Excel | Add `--force-reload` |
| Generalized run with one person | Works, but generalized is mainly useful with **2+** people registered |

For code or environment issues, contact the team lead with the **full terminal error** and the exact command you ran.

---

## Reference: existing example

| Item | Value |
|------|--------|
| Data folder | `PilotPhaseC/data/Ring_01_PID_104/` |
| Config key | `ring01_pid104` |
| Example command | `.venv/bin/python -m PilotPhaseC.src.train_classifier --participant ring01_pid104 --lineage all_continuous` |

More background: `PilotPhaseC/README.md`.
