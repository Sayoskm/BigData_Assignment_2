# Assignment 2 - Predicting GitHub User Activity

A heterogeneous GNN (relational deep learning) that predicts, for each test user, which of 6
activities they will do during 2026/11/22 00:00 – 11/29 00:00 (Taiwan time).

All of my own code is in one notebook, `run.ipynb`. It calls the TA's `scripts/preprocess.py`
(to build the 8 tables) and `scripts/evaluate.py` (to score and check the format), which are
used unchanged.

## Data Source

GH Archive: https://www.gharchive.org/

The data is used for educational purposes within this course only.

## Files

```
.
├── .gitignore
├── README.md
├── SCHEMA.md
├── requirements.txt
├── run.ipynb
├── scripts/
│   ├── cohort.txt
│   ├── evaluate.py
│   └── preprocess.py
└── submission/
    ├── sample_submission.csv
    └── test_input.csv
```

## How to run

Tested with Python 3.13.1 on Windows, CPU only (no GPU needed).
Versions used: torch 2.15.0+cpu, torch_geometric 2.8.1, pandas 3.1.1, numpy 2.5.4,
pyarrow 26.0.1, matplotlib 3.11.2.

1. Install the packages (plus Jupyter to run the notebook):

   ```bash
   pip install -r requirements.txt
   pip install jupyter
   ```

2. Open `run.ipynb` **from the repository root** (it imports `scripts/evaluate.py` and uses
   relative paths) and run all cells in order.

The notebook does everything:

| Section | What it does | Time (approx.) |
|---|---|---|
| 1 | Settings: data period (`START`, `END`), prediction time, output file name, paths | – |
| 2 | Downloads the GH Archive hourly files into `work/raw/` (existing files are skipped, so it can be rerun) | depends on network |
| 3 | Runs `scripts/preprocess.py --raw work/raw --out work/data` | ~6 min |
| 4–6 | Data checks, weekly activity labels, the copy-last-week baseline | < 1 min |
| 7 | Snapshot, graph building, GNN, walk-forward validation | ~7 min |
| 8 | Trains the final model, writes **`work/outputs/output.csv`**, checks the format with `evaluate.py` | ~10–13 min |
| 9 | Extra experiments (ablations); not needed for the output | ~30 min |

To only reproduce the output, run sections 1–8; section 9 can be skipped.

## Data period

Hourly files from **2026/08/30 00:00 to 2026/11/15 00:00 (Taiwan time)**
= 2026-08-29 16:00 – 2026-11-14 16:00 UTC, 11 full weeks, 1,848 hourly files.
The period is fixed in the notebook (`START`, `END`).

All dates below are Taiwan time. The notebook works in UTC, so a cutoff written there as
`2026-09-05 16:00` is 2026/09/06 00:00 here.

## Method

1. **Snapshot**: for a cutoff time t, keep only rows whose row time is before t, and hide the
   later-known columns (`merged_at`, `merged_by_id`, `closed_at`, `state_reason`,
   `dismissed_at`) if they are not before t.
2. **Graph**: every table row is a node (8 node types) and every foreign key is an edge type
   (20 types, 39 with reverse edges). Node features are flags and time encodings: for each
   time column, d = days before t gives log(1 + d), exp(−d/7), exp(−d/30) (missing → 365 days),
   plus the day of week as sin / cos for comments, reviews and pushes.
3. **Model**: per-node-type linear encoder (64 dims), 2 layers of `HeteroConv` with one
   `SAGEConv` per edge type (sum aggregation), residual + LayerNorm, and an MLP head on user
   nodes with 6 outputs.
4. **Training**: one graph per weekly cutoff t = 9/6, 9/13, …, 11/1 (9 graphs). The label is
   the activity in [t + 7 days, t + 14 days), the same 7-day gap as the real task (data ends
   11/15, test week starts 11/22). BCE loss on cohort users, Adam (lr 3e-3), 40 epochs, seed 0.
5. **Thresholds**: one threshold per activity, chosen to maximize F1 on the most recent
   labeled week (11/8 – 11/15 for the final run).
6. **Prediction**: build the graph at t = 11/15 and predict 11/22 – 11/29.

## Validation

Walk-forward over 4 weeks: pretend the data ends one week earlier each time and repeat the
full pipeline (training graphs, thresholds, prediction). Score = (macro-F1 + micro-F1) / 2
from `evaluate.py`.

| Data ends | Training cutoffs | Predicted week | GNN | copy_last_week |
|---|---|---|---|---|
| 10/11 | 9/6 – 9/27 (4) | 10/18 | 0.462 | 0.421 |
| 10/18 | 9/6 – 10/4 (5) | 10/25 | 0.378 | 0.353 |
| 10/25 | 9/6 – 10/11 (6) | 11/1 | 0.322 | 0.281 |
| 11/1 | 9/6 – 10/18 (7) | 11/8 | 0.358 | 0.290 |
| | | **mean** | **0.380** | 0.336 |

Ablations (section 9, mean over the same 4 weeks):

| Model | Change | Mean score |
|---|---|---|
| GNN | – | 0.380 |
| GNN_user_only | only edges between events and users (no cross-table edges) | 0.377 |
| GNN_thr_fixed | thresholds tuned once in the first run only | 0.376 |
| GNN_no_time | all time features set to 0 | 0.346 |
| GNN_mean | mean instead of sum aggregation | 0.305 |
| GNN_thr_0.5 | all thresholds = 0.5 | 0.281 |

## Output

`work/outputs/output.csv`: one row per test user (4,465 rows) with columns
`UserID, open_pr, open_issue, issue_comment, pr_review, pr_review_comment, push` (0/1).
The format is checked with `scripts/evaluate.py` at the end of section 8.
Rename it to `output_{studentID}.csv` for submission.

## Reproducibility

- The data period is fixed and only `preprocess.py` is used to build the tables.
- `torch.manual_seed(0)` is set before training and everything runs on CPU, so running the
  notebook again gives the same `output.csv`.
