# Stress classification from a smart ring

## Problem

Classify a wearer's stress state from physiological and motion signals recorded by a smart ring, using self-reported surveys as ground truth. The model must be small enough to run on the ring's microcontroller.

## Classes

| Class | Meaning |
|---|---|
| `relax` | Calm, low arousal |
| `eustress` | Positive, motivating stress |
| `distress` | Negative stress |
| `eustress+distress` | Both reported at the same time |

We also study the coarser 2-class version: `good` (`relax` + `eustress`) vs `bad` (`distress` + `eustress+distress`).

## Data

- Each sample is about 5 seconds of active sensing at ~25 Hz, captured roughly every 20 seconds.
- **Slow-moving signals:** skin temperature, LED signal levels, humidity.
- **Fast-moving signals:** heart rate, heart rate variability and related pulse features.
- **Accelerometer:** motion and activity, used to identify movement and periods when the ring is off the finger.
- **Labels** come from periodic surveys. One survey answer is carried forward over the following interval (up to 60 minutes), so labels are only loosely tied to the moment they describe.

## Key challenges

- Few independent labelled units per person (tens of survey intervals).
- Labels are noisy in time, and class mixes differ by person.
- Slow sensor levels differ between sessions and can look like class signal when they are not.
- Motion and ring removal can distort signals within a class.

## What this repo is for

This branch holds the project plan and overview. The current focus is a sprint to understand the data: whether labels can be trusted, what a pseudo-labelling algorithm should look like, and whether a 2-class framing is clearer than 4 classes. See [`docs/SPRINT_PLAN.md`](docs/SPRINT_PLAN.md).
