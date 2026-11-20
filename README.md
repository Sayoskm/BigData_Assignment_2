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
run.ipynb                  # all of my code: download -> preprocess -> model -> output
requirements.txt
scripts/preprocess.py      # TA's script
scripts/evaluate.py        # TA's script
scripts/cohort.txt
submission/test_input.csv  # official test users
SCHEMA.md                  # TA's description of the 8 tables
work/                      # created when running (not in git)
  raw/                     # GH Archive hourly files
  data/db/                 # the 8 tables written by preprocess.py
  outputs/output.csv       # final prediction
```

## How to run

Tested with Python 3.13.1 on Windows, CPU only (no GPU needed).
Versions used: torch 2.15.0+cpu, torch_geometric 2.8.1, pandas 3.1.1, numpy 2.5.4.

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
| 1 | Settings: data period, prediction time, paths | – |
| 2 | Downloads GH Archive hourly files into `work/raw/` (existing files are skipped) | depends on network |
| 3 | Runs `scripts/preprocess.py --raw work/raw --out work/data` | ~6 min |
| 4–6 | Data checks and the copy-last-week baseline | < 1 min |
| 7 | Snapshot, graph building, GNN, walk-forward validation | ~7 min |
| 8 | Trains the final model and writes **`work/outputs/output.csv`**, then checks the format with `evaluate.py` | ~10–13 min |
| 9 | Extra experiments (ablations); not needed for the output | ~30 min |

To only reproduce the output, run sections 1–8; section 9 can be skipped.

## Data period

Hourly files from **2026-08-29 16:00 to 2026-11-14 16:00 UTC** (= 2026/08/30 00:00 –
11/15 00:00 Taiwan time, 11 full weeks, 1,848 hours). The period is fixed in the notebook
(`START`, `END`).

## Reproducibility

- The data period is fixed and only `preprocess.py` is used to build the tables.
- `torch.manual_seed(0)` is set before training and everything runs on CPU, so running the
  notebook again gives the same `output.csv`.

## Method (short)

1. **Snapshot**: for a cutoff time t, keep only rows whose row time is before t, and hide the
   later-known columns (`merged_at`, `merged_by_id`, `closed_at`, `state_reason`,
   `dismissed_at`) if they are not before t.
2. **Graph**: every table row is a node and every foreign key is an edge type (plus reverse
   edges). Node features include flags and time encodings (days before t, day of week).
3. **Model**: per-node-type linear encoder, 2 layers of heterogeneous GraphSAGE (sum
   aggregation), and an MLP head on user nodes with 6 outputs.
4. **Training**: one graph per weekly cutoff; the label is the week starting 7 days after the
   cutoff (the same 7-day gap as the real task).
5. **Thresholds**: tuned per activity to maximize F1 on the most recent labeled week.
6. **Validation**: walk-forward over 4 weeks (predicting 10/17, 10/24, 10/31, 11/7), each run
   with the same steps as the final prediction.
