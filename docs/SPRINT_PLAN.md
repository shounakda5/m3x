# Phase C data-pattern study: week 1 plan

## Goal

Understand how every model input behaves per class, over time and across surveys, so we can answer three questions by the end of the sprint (1-2 weeks):

1. **Can we trust the labels?** Do data trends support them?
2. **What should a pseudo-label algorithm look like?**
3. **Is the signal clearer as a 2-class problem (`good` vs `bad`) than as a 4-class problem? By how much?**

Background: `PERFORMANCE_REPORT.md` (sections 3-4). Ring 4 currently performs at chance, so every finding must state whether it also holds for Ring 4.

## Data

- 5 seconds of active data at ~25 Hz, collected every ~20 s.
- Slow features: temperature, LED signals, humidity.
- Fast features: HR, HRV, etc.
- Accelerometer: motion and activity.
- Datasets already exist at `data/dataset/ring0X_pidXXX/all_continuous_*.npz`. No new extraction is needed.
- 2-class definition: `good` = relax + eustress; `bad` = distress + eustress+distress.

## Ground rules

- Analyse **each ring separately** first (Rings 1, 3, 4), then pool.
- Report every class comparison as **both 4-class and 2-class**.
- The unit of evidence is the **survey interval**, not the window. Windows in one survey are not independent, so give effect sizes with survey-level confidence intervals.
- Log every finding in one shared table: feature or group, ring, claim, effect size, 4-class vs 2-class, figure, confidence (high / medium / low).
- One notebook per workstream in `m3x/PilotPhaseC/analysis/`. Keep code in the notebook; do not split it into modules.

## Who works on what

| Person | Workstream | Starts Day 1 with | Waiting on anyone to start? |
|---|---|---|---|
| Tenor | **A**: fast features, transitions, 2-class vs 4-class, label trust (feature study) | A1: per-class fast-feature distributions | No |
| Anisha | **B**: slow features, accelerometer, wear and artifacts (feature study) | B1: per-class slow-feature distributions | No |
| Yaxian | **C**: temporal trends and sampling rate | C1: 10/30/60-minute aggregation | No |

Tenor and Anisha together cover the feature study, with comparable volume: A is heavier on interpretation (label trust, 2 vs 4 class), B on data building (motion bins, anomaly catalogue). Yaxian runs the temporal study independently.

## Three workstreams

| | Workstream | Main question |
|---|---|---|
| **A** | Fast features, class transitions, 2-class vs 4-class, label trust | What separates the classes, and do the labels hold up? |
| **B** | Slow features, accelerometer, wear and artifacts | Does motion or taking off the ring distort the data within a class? |
| **C** | Temporal trends and sampling rate | How do features change over 10/30/60 minutes, and how much data can we drop? |

### A: fast features, transitions, label trust

Tasks:

1. Per-class distributions of fast features, per ring, raw and per-person-normalised. Rank features by effect size for each class pair.
2. 2-class vs 4-class: for each feature, compare effect size and one-vs-rest AUC. Summarise the gain per feature group as a number.
3. Class transitions: align signals to each survey boundary where the class changes (±30 min). Compare against same-class boundaries as a control. Does any signal change appear?
4. Label trust: how fast does class separability fall as windows get further from the survey time?

Deliverables:

| Day | Deliverable |
|---|---|
| 2 | Ranked feature table (effect sizes per class pair and ring) |
| 4 | 2-class vs 4-class comparison with one summary chart and a headline number |
| 5 | Transition plots with a written verdict: do transitions leave a signal footprint? |
| 5 | Draft label-trust verdict per ring (trustworthy / partly / not) with evidence |

### B: slow features, motion, wear

Tasks:

1. Per-class distributions of slow features. Split within-person drift from class differences. (Pilot B lost ~25 points of accuracy when the slow component was removed, so quantify how much class information is really there.)
2. Build a per-window motion intensity from the accelerometer. Define low / medium / high activity bins and a "no motion / likely off finger" flag. Report the activity mix per class and ring.
3. Within a class, compare feature distributions in low vs high motion windows. Identify which features are corrupted by motion and which are robust.
4. Catalogue anomalous surveys (poor signal, ring not worn, extreme outliers) with a reason for each. Summarise Ring 4 data quality.

Deliverables:

| Day | Deliverable |
|---|---|
| 2 | Slow-feature class profiles with the drift vs class-signal split |
| 3 | Activity bins and per-class activity mix |
| 5 | Motion-conditioned feature comparison: corrupted vs robust features |
| 5 | Anomalous survey list with reasons, and Ring 4 data-quality summary |

### C: temporal trends and sampling rate

Tasks:

1. Aggregate every feature and feature group over 10, 30 and 60 minutes (windows must not cross a survey boundary). Compare per-class trends and variance within vs between windows, for slow and fast groups.
2. Within-episode dynamics: how features evolve from the start to the end of a survey interval, including the up-to-60-minute label carry-forward.
3. Sampling-rate study: drop rows at several rates (keep 1 in 2 / 4 / 8, and random dropout of 25 / 50 / 75%), recompute the step 1 results, and measure how class separability, trend shape and variance change against full data.
4. Result: for each feature group, the lowest sampling rate and shortest window at which it still tells its story.

Deliverables:

| Day | Deliverable |
|---|---|
| 2 | Aggregation in the notebook and first 10/30/60-minute trend plots |
| 3 | Within-episode dynamics summary |
| 5 | Sampling-rate results: feature-survival chart and rate-vs-window grid |

## Dependencies

| Deliverable | Needs | Note |
|---|---|---|
| A2, B1, C1 (Day 2) | Nothing | All three start on Day 1 |
| A: 2-class vs 4-class | A: ranked feature table | Same workstream |
| B: motion-conditioned comparison | B: activity bins | Can use the A ranking to prioritise features, but does not wait for it |
| A: label-trust verdict | B: anomalous survey list | Draft uses all surveys; final repeats without flagged surveys |
| C: sampling-rate results | C: aggregation | Same pipeline on degraded input |
| Week 2 pseudo-label design | A label-trust verdict, B anomaly list, C within-episode dynamics | |

```
Day 1-2   A1 -> A ranked table ──┐ (soft: helps B prioritise)
          B1 -> B profiles       │
          C1 -> C trend plots    │
Day 3-4   A: 2 vs 4 class        │
          B: activity bins ──────┘
          C: within-episode dynamics
Day 5     A: label-trust draft <── B: anomaly list (interim: all surveys)
          B: motion comparison
          C: sampling-rate study <── C1 aggregation
```

### How nobody gets blocked

- **Day 1:** Tenor, Anisha and Yaxian each start on tasks that need no input from anyone else (A1, B1, C1).
- **Only two cross-person hand-offs exist.** Both are soft, and each has an interim version so work continues without waiting:
  - A's ranked feature table helps B prioritise. B starts with all features and re-prioritises when the table arrives (Day 2).
  - B's anomaly list feeds A's label-trust verdict. A drafts using all surveys and repeats without flagged surveys once the list arrives (Day 5).
- **If someone finishes early:** pick up the next task in their own workstream. If the workstream is finished, run its checks on Ring 4 separately, or repeat them as 2-class.
- **If someone is stuck for more than half a day:** raise it at the Day 2 sync, or immediately in the team channel.

## Week 1 timeline

| Day | What happens |
|---|---|
| 1 | 30-min kickoff (ground rules, findings table). Everyone starts task 1 |
| 2 | First deliverable from each workstream. 15-min sync |
| 3-4 | Second-tier tasks |
| 5 | Remaining deliverables land. 60-min review; set week 2 |

## Week 2 preview

- Design the pseudo-label algorithm from the label-trust verdict, anomaly list and temporal dynamics.
- Check that week 1 trends hold on a held-out person (train on two, test on the third).
- Draft answers to the three goal questions, with the evidence from the shared findings table.
