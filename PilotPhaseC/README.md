# Pilot Phase C

A new, ongoing phase of the project: unlike the original pipeline (`data/` +
root-level `train.py`/`train_cnn.py`) and the Pilot B cohort (`pilot_b/`),
this phase's incoming data arrives **already labeled**. There is no
survey-matching or burst-labeling step -- the pipeline here is just:

```
incoming batch(es)  -->  validated, combined dataset snapshot  -->  train  -->  export
```

New data is expected to keep arriving over the coming weeks, with a new
model version retrained as it does. Everything in this directory is
self-contained: its own data, its own code, its own results, its own
config -- it does not modify the original pipeline or the Pilot B cohort.

## Layout

```
PilotPhaseC/
├── config.py                  # paths, label taxonomy (CLASSES), hyperparameters
├── data/
│   ├── incoming/
│   │   └── <batch_id>/        # one raw drop, exactly as received -- never edited in place
│   │       ├── manifest.json
│   │       └── *.csv
│   └── dataset/
│       └── <version>/         # frozen, combined snapshot used to train that version
│           ├── dataset.csv
│           └── manifest.json  # which incoming batches went into this snapshot
├── src/
│   ├── loader.py               # ingest + validate + combine incoming batches
│   └── train.py                 # train against a frozen dataset snapshot
└── results/
    └── <version>/
        ├── <model files>
        └── manifest.json        # dataset version, git commit, hyperparams, metrics
```

## Adding a new data batch

1. Drop the raw files, untouched, into `data/incoming/<batch_id>/` -- a
   descriptive, date-stamped name works well, e.g. `2026-10-21_batch1`.
2. Add a `manifest.json` alongside them describing the batch:

   ```json
   {
     "source": "where this batch came from",
     "date_range": ["2026-10-14", "2026-10-21"],
     "participants": ["..."],
     "n_rows": 1234,
     "label_scheme": "describe how/where the labels in this batch were assigned"
   }
   ```
3. Never edit files inside an existing `incoming/<batch_id>/` after the
   fact -- if corrected data arrives, add it as a new batch instead, so any
   dataset snapshot already cut from the old batch stays reproducible.

## Cutting a new dataset snapshot

Combine whichever batches you want into a frozen, versioned snapshot:

```bash
python -m PilotPhaseC.src.loader --batches 2026-10-21_batch1 2026-11-04_batch2 --version v1
```

This writes `data/dataset/v1/dataset.csv` plus a `manifest.json` recording
exactly which batches went into it. Snapshots are immutable once cut --
cutting `v2` later (e.g. after a new batch arrives) does not touch `v1`, so
any model trained against `v1` stays reproducible.

## Training a new model version

```bash
python -m PilotPhaseC.src.train --dataset-version v1 --output-version v1
```

Writes `results/v1/` with a `manifest.json` recording the dataset version,
git commit, hyperparameters, and metrics for that run -- so "what data
trained this model" always has a one-file answer.

**Before the first real batch lands**, two things in `src/loader.py` and
`src/train.py` are placeholders and need filling in once the actual data
format is known:

- `loader.py`'s `REQUIRED_COLUMNS` -- the real pre-labeled schema (label
  column, participant/session identifiers, timestamps, etc).
- `config.py`'s `CLASSES` -- the real label taxonomy.
- `train.py`'s feature-extraction / model-selection block -- depends on
  whether the data is raw time-series (reuse `src/models/cnn_model.py`,
  same pattern as `pilot_b/`) or tabular features (reuse
  `src/models/{svm,rf,xgb}_model.py`, same pattern as the root pipeline).
