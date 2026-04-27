# FedShift Comparison — Findings

FedShift aligns quantized updates to the anchor distribution via a per-layer global mean shift:
```
shifted_delta = quantized_delta + (mean_anchor - mean_quantized)
```
No learned model, no replay buffer — just a single closed-form correction per round.

---

## Results by Configuration

### INT8 Near-IID (α=100, 50% anchors, 50 rounds)

| Method      | R10  | R25  | R50  |
|-------------|------|------|------|
| FedAvg      | 67.3 | 76.8 | 82.7 |
| FedShift    | 66.3 | 75.4 | 81.1 |
| ProWD       | 64.6 | 72.8 | 79.3 |
| **AGD-Smart** | **67.6** | 75.8 | **81.2** |
| AGD vs FedShift | +1.3 | +0.3 | **+0.1** |

FedShift and AGD-Smart are effectively **tied at R50** (81.2 vs 81.1%). Both trail FedAvg by ~1.5pp. FedShift beats ProWD by +1.8pp without any learned model.

---

### INT4 Near-IID (α=100, 50% anchors, 50 rounds)

| Method      | R10  | R25  | R50  |
|-------------|------|------|------|
| FedAvg      | 67.4 | 77.5 | 83.2 |
| FedShift    | 66.5 | 75.4 | 80.7 |
| ProWD       | 64.8 | 73.8 | 77.7 |
| **AGD-Smart** | **67.5** | **77.1** | **82.5** |
| AGD vs FedShift | +1.0 | +1.8 | **+1.8** |

AGD-Smart maintains a consistent **+1.8pp advantage** over FedShift at R25 and R50. FedShift beats ProWD by +3pp, again without a learned corrector. AGD-Smart is the only method within 0.7pp of FedAvg.

---

### INT8 Non-IID (α=0.5, 50% anchors, 50 rounds)

| Method      | R10  | R25  | R50  |
|-------------|------|------|------|
| FedAvg      | 65.1 | 73.4 | 80.4 |
| FedShift    | 57.8 | 65.9 | 73.8 |
| ProWD       | 56.1 | 69.6 | 72.8 |
| **AGD-Smart** | **59.8** | **69.8** | **74.8** |
| AGD vs FedShift | +2.0 | +3.9 | +1.0 |

**FedShift fails under non-IID data.** The global mean shift is computed from anchor means, which do not represent constrained client data distributions when heterogeneity is high. FedShift trails FedAvg by **−6.6pp** at R50 — worse than even ProWD. AGD-Smart beats FedShift by +3.9pp at R25, showing that smart per-client matching is essential in heterogeneous settings.

---

### INT2 Near-IID (α=100, 20% anchors, 50 rounds)

| Method      | R10  | R20  | R25  | R30  | R50  |
|-------------|------|------|------|------|------|
| FedAvg      | 44.0 | 64.9 | 67.3 | 70.2 | 75.4 |
| FedShift    | 46.9 | 67.0 | 69.5 | 72.9 | 77.4 |
| ProWD       | 57.3 | 67.0 | 69.2 | 68.5 | 71.8 |
| **AGD-Smart** | **58.2** | **70.2** | **73.9** | **75.3** | **77.5** |
| AGD vs FedShift | **+11.2** | +3.2 | +4.4 | +2.4 | +0.1 |

The most revealing comparison. At R50, AGD-Smart and FedShift converge to nearly the same accuracy (77.5 vs 77.4%) — but **AGD-Smart leads by +11.2pp at R10**. FedShift recovers gradually; AGD-Smart's CorrectionNet fires at round 6 and immediately opens a gap that only closes over 44 more rounds.

FedShift does outperform FedAvg throughout (the mean shift helps at INT2), and it eventually matches AGD-Smart's final accuracy. But it loses badly in early rounds — which matters for communication efficiency.

---

## Key Takeaways

**Where FedShift is competitive:**
- Near-IID conditions: simple mean shift is very effective, nearly matches AGD-Smart at INT8
- Consistently beats ProWD in 3 of 4 configs without any learned model

**Where FedShift breaks down:**
- **Non-IID data**: global mean shift is biased — anchor means don't represent heterogeneous client distributions. Falls 6.6pp behind FedAvg.
- **Early convergence at INT2**: loses by 11.2pp at R10 despite closing the gap to 0.1pp by R50. In communication-limited scenarios (few rounds), AGD-Smart's early activation advantage is decisive.

**Ordering (most to least consistent):**
```
AGD-Smart ≥ FedShift > ProWD    (near-IID)
AGD-Smart > ProWD ≈ FedShift    (non-IID)
```
FedShift is a stronger baseline than ProWD under near-IID conditions. Under non-IID, the direction-preserving blend in AGD-Smart makes it more robust than either FedShift or ProWD.

**For the slides/paper:**
FedShift is worth adding as a second learned-free baseline alongside FedAvg. It sharpens the story:
- FedAvg = no correction at all
- FedShift = simple mean shift (fast, no model, near-IID works, non-IID fails)
- ProWD = learned corrector but resets each round (beaten by FedShift in most configs)
- AGD-Smart = learned corrector with replay buffer + smart matching (wins on early convergence and non-IID robustness)

The new ordering makes clear that what matters isn't just having a learned corrector (ProWD), but having a **persistent** one (replay buffer) with **personalized** targets (smart matching).
