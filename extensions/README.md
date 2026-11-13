# QFLARE bonus studies

These are two separate comparisons using the same **max-stall** attack method. They do not change the required root-level `attack.py` or any released evaluator file. There is no four-class extension in this submission.

## Method and fairness

`scheduled_attack.py` derives its update from the max-stall approach in `max_stall_dynamic.py`: in each selected round, both compromised clients perturb their honest circuit deltas at the permitted radius in the direction opposite the mean honest delta. The update formula is identical in all six settings. Only the list of selected rounds changes. No random generator is used by the attack.

The runner calls the released `challenge.prepare` and `challenge.train` with public seeds **11** and **29**, all three defenses, 60 rounds, two compromised clients, and the released perturbation limit. For every seed and defense, the clean predictions come from `outputs/result.json`, whose rules and release-file hashes are checked against the current repository. The runner checks the test IDs and labels against a fresh `prepare(seed)` call, then trains each attacked setting with that same prepared split, initialization, client partition, and training budget. Each saved JSON includes all predictions, labels, selected and applied rounds, source hashes, and summary.

Macro-F1 is computed separately for clean and attacked predictions. `Drop (pp) = 100 × (clean F1 − attacked F1)`; a negative number means the attack improved the classifier in that case. The mean is over all six cases. Values below are local benchmark observations, not organizer-awarded bonus points.

## Attack timing: 30 rounds in each setting

`first30` uses rounds 0–29, `last30` uses 30–59, and `alternating30` uses 0, 2, …, 58.

| Setting | Seed | Defense | Clean F1 | Attacked F1 | Drop (pp) |
|---|---:|---|---:|---:|---:|
| alternating30 | 11 | fedavg | 0.977767 | 0.988888 | -1.112072 |
| alternating30 | 11 | median | 0.977778 | 0.988888 | -1.110974 |
| alternating30 | 11 | trimmed_mean | 0.988888 | 0.988888 | +0.000000 |
| alternating30 | 29 | fedavg | 0.933201 | 0.899690 | +3.351098 |
| alternating30 | 29 | median | 0.933300 | 0.921981 | +1.131897 |
| alternating30 | 29 | trimmed_mean | 0.933300 | 0.933201 | +0.009901 |
| **alternating30 mean** | | | | | **+0.378308** |
| first30 | 11 | fedavg | 0.977767 | 0.988888 | -1.112072 |
| first30 | 11 | median | 0.977778 | 0.988888 | -1.110974 |
| first30 | 11 | trimmed_mean | 0.988888 | 0.988888 | +0.000000 |
| first30 | 29 | fedavg | 0.933201 | 0.922136 | +1.106568 |
| first30 | 29 | median | 0.933300 | 0.933300 | +0.000000 |
| first30 | 29 | trimmed_mean | 0.933300 | 0.922136 | +1.116469 |
| **first30 mean** | | | | | **-0.000001** |
| last30 | 11 | fedavg | 0.977767 | 0.966630 | +1.113721 |
| last30 | 11 | median | 0.977778 | 0.966630 | +1.114819 |
| last30 | 11 | trimmed_mean | 0.988888 | 0.977767 | +1.112072 |
| last30 | 29 | fedavg | 0.933201 | 0.888393 | +4.480853 |
| last30 | 29 | median | 0.933300 | 0.922213 | +1.108778 |
| last30 | 29 | trimmed_mean | 0.933300 | 0.933333 | -0.003294 |
| **last30 mean** | | | | | **+1.487825** |

The late schedule had the largest mean drop, but individual cases vary substantially. In particular, the early and alternating schedules improved the attacked model in two seed-11 cases.

## Number of attack rounds: evenly spaced budgets

For budget `b`, selected rounds are `floor((i + 0.5) × 60 / b)` for `i = 0, …, b−1`. This rule is unchanged for budgets 10, 20, and 30.

| Setting | Seed | Defense | Clean F1 | Attacked F1 | Drop (pp) |
|---|---:|---|---:|---:|---:|
| even10 | 11 | fedavg | 0.977767 | 0.988888 | -1.112072 |
| even10 | 11 | median | 0.977778 | 0.988888 | -1.110974 |
| even10 | 11 | trimmed_mean | 0.988888 | 0.988888 | +0.000000 |
| even10 | 29 | fedavg | 0.933201 | 0.921981 | +1.121996 |
| even10 | 29 | median | 0.933300 | 0.933300 | +0.000000 |
| even10 | 29 | trimmed_mean | 0.933300 | 0.933300 | +0.000000 |
| **even10 mean** | | | | | **-0.183508** |
| even20 | 11 | fedavg | 0.977767 | 0.988888 | -1.112072 |
| even20 | 11 | median | 0.977778 | 0.988888 | -1.110974 |
| even20 | 11 | trimmed_mean | 0.988888 | 0.988888 | +0.000000 |
| even20 | 29 | fedavg | 0.933201 | 0.944383 | -1.118126 |
| even20 | 29 | median | 0.933300 | 0.933300 | +0.000000 |
| even20 | 29 | trimmed_mean | 0.933300 | 0.933300 | +0.000000 |
| **even20 mean** | | | | | **-0.556862** |
| even30 | 11 | fedavg | 0.977767 | 0.966563 | +1.120333 |
| even30 | 11 | median | 0.977778 | 0.966630 | +1.114819 |
| even30 | 11 | trimmed_mean | 0.988888 | 0.966563 | +2.232405 |
| even30 | 29 | fedavg | 0.933201 | 0.887500 | +4.570139 |
| even30 | 29 | median | 0.933300 | 0.922136 | +1.116469 |
| even30 | 29 | trimmed_mean | 0.933300 | 0.944438 | -1.113719 |
| **even30 mean** | | | | | **+1.506741** |

The 30-round budget had the largest mean drop. The relationship is not monotonic: 20 rounds scored below 10 rounds, and the 30-round attack improved one seed-29 trimmed-mean case.

## Reproduce

From the repository root in Ubuntu, using Python 3.11 or 3.12 and the pinned `requirements.txt`:

```bash
source .venv/bin/activate
python score.py outputs/result.json --submission attack.py
python extensions/run_comparisons.py --rerun
```

The second command regenerates the six JSON files in `extensions/timing/` and `extensions/budget/`. It takes several minutes on a CPU. To rerun one setting, append `--setting timing_last30` or another key from `scheduled_attack.py`. Without `--rerun`, completed cases are preserved and the runner resumes missing cases. The required binary submission remains the root `attack.py` plus `outputs/result.json`; these bonus studies are separate.
