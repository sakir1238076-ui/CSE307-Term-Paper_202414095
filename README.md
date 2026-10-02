# CSE307-Term-Paper_202414095

# CSE-307 Track 3 — Deadlock Early Risk Classifier

**Name:** Md. Nasimul Goni  
**Roll:** 202414095  
**Course:** CSE-307 Operating Systems — Part B (Term Paper)  
**Submitted to:** Lec Khaled Hasan Irfan

## Overview
Banker's Algorithm and Resource Allocation Graph (RAG) cycle detection only confirm that a state is unsafe once it has been reached. This project asks whether a lightweight **Decision Tree**, looking only at a *partial* request sequence, can flag a risky trajectory **earlier** than the Banker's check.

## What is implemented
| File | Purpose |
|---|---|
| `src/banker.py` | Banker's Algorithm: Need matrix, safety check, resource-request check. **Ground truth** for every label. |
| `src/rag.py` | RAG construction and DFS cycle detection. |
| `src/workload.py` | Synthetic workload generator: `RANDOM` and the shifted `LATE_UNSAFE` workload (safe first half, unsafe-pushing second half). |
| `src/ml.py` | Prefix features, `DecisionTreeClassifier(max_depth=4)`, risk curve, rule-based explanation generator (bonus). |
| `src/experiment.py` | Training, evaluation, metrics (lead time, FPR, FNR, confidence). |
| `src/plotting.py` | Figures written to `results/`. |
| `tests/` | Unit tests (textbook Banker's example, RAG cycles, workload shift, no-leakage check). |
| `run_experiment.py` | Runs the whole pipeline. |

### Design in short
- **Sequence:** `T = 4 × processes` requests, granted one by one (without Banker's blocking them) so the trajectory can be watched. After every step the state is labelled safe/unsafe by Banker's.
- **Learned component:** features come only from the state *so far* (utilization, scarcest free resource, request pressure, max per-resource pressure). Label = "this trajectory eventually becomes unsafe". Trained on `RANDOM` sequences, then tested on both `RANDOM` (no shift) and `LATE_UNSAFE` (shift).
- **Shift:** in `LATE_UNSAFE` the first half only picks requests that keep the state safe; from the halfway step it picks requests that push the state unsafe. Tests confirm no sequence is unsafe before the shift.
- **System sizes:** Config A (4 processes × 2 resources), B (8 × 4), C (12 × 6). 1000 training and 300 test sequences per configuration. Seed is fixed, so results are reproducible.

### Metrics
- **Lead time** = (step Banker's marks the state unsafe) − (step the classifier first flags risk), averaged over unsafe sequences the classifier flagged. Positive = earlier than Banker's, negative = later.
- **FPR** = safe sequences flagged risky at any step / all safe sequences. **FNR** = unsafe sequences never flagged / all unsafe sequences.
- **Early rate** = share of unsafe sequences flagged strictly before Banker's.
- **Confidence** (bonus) = max(p, 1−p) at each step, averaged separately over correct and wrong step predictions.

## How to run
```bash
pip install -r requirements.txt
python run_experiment.py        # writes tables, figures and explanations to results/
python -m pytest -q             # run the unit tests
```

## Results (LATE_UNSAFE shift, threshold 0.5)
| Config | Procs × Res | Unsafe / 300 | Avg lead time | FPR | FNR | Early rate | Conf. correct / wrong |
|---|---|---|---|---|---|---|---|
| A | 4 × 2 | 48 | +7.6 steps | 0.944 | 0.021 | 0.979 | 68.7 / 71.8 |
| B | 8 × 4 | 36 | +1.6 steps | 0.617 | 0.444 | 0.472 | 56.4 / 68.3 |
| C | 12 × 6 | 59 | −1.1 steps | 0.241 | 0.746 | 0.000 | 64.4 / 59.8 |

Full numbers (including the no-shift `RANDOM` workload) are in `results/metrics_summary.csv`. Figures: `results/lead_time.png`, `error_rates.png`, `detection_rate.png`, `confidence_calibration.png`, `risk_over_time.png`. Example step-by-step explanations are in `results/explanations.txt`.

## Analysis
- **Banker's Algorithm blocks the unsafe request exactly when it happens**, so its detection delay is zero by definition.
- **Lead time shrinks as the system grows** (+7.6 → +1.6 → −1.1 steps). With more processes and resources, one risky request changes aggregate features such as utilization by a smaller amount, so the classifier needs more steps to notice.
- **The early warning in Config A is bought with false alarms.** It flags 94% of safe sequences, so it is almost always warning. Config C is the opposite: few false alarms (24%) but it misses 75% of unsafe sequences. No configuration gets both rates low.
- **Risk rises only weakly after the shift** (mean risk 0.64 → 0.70 in A, 0.48 → 0.54 in B, 0.36 → 0.41 in C): the aggregate features barely separate "about to become unsafe" from "safe but busy".
- **Confidence is not well calibrated.** In Configs A and B the model is *more* confident when it is wrong; only Config C goes the expected direction (64 vs 60).
- **Conclusion:** a depth-4 tree on aggregate features cannot replace Banker's. It needs per-process information (e.g. how many processes can still finish) to give a reliable early signal.

## Limitations
- Requests in the generator are granted without Banker's blocking them, so trajectories show how an unprotected system would evolve.
- The RAG is exact for single-instance resources only; for multi-instance resources a cycle is necessary but not sufficient, so Banker's remains the ground truth.
- Synthetic workloads only; results depend on the generator settings in `src/workload.py` (e.g. `DEMAND_LOAD`).

## AI Disclosure
I used an AI assistant (Claude) to help write and structure the code in this repository (generator, experiment pipeline, plotting, tests), to debug it, and to polish the wording of this README and the report. I ran the experiments, and the numbers above are the outputs of this code. I am responsible for understanding and explaining every part of the submission.
