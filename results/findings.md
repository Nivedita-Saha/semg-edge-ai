# Findings — model compression for sEMG gesture recognition

## The headline

A 1D-CNN was trained on Ninapro DB5 (subject 1, 17 hand gestures, Myo
armband) and compressed using the three techniques named in the CSPR002
advert: post-training int8 quantisation, magnitude pruning, and knowledge
distillation. Every model was converted to TensorFlow Lite and benchmarked
through an identical pipeline.

| Model | Params | Size (KB) | Latency (ms) | Accuracy |
|---|---|---|---|---|
| Baseline | 18,545 | 79.9 | 0.012 | 74.13% |
| Baseline + int8 | 18,545 | 31.8 | 0.005 | 74.13% |
| Pruned (48%) | 18,545 | 80.1 | 0.012 | 75.46% |
| **Pruned + int8** | 18,545 | **32.1** | **0.005** | **75.62%** |
| Student (distilled) | 5,697 | 31.1 | 0.005 | 71.64% |
| **Student + int8** | 5,697 | **18.3** | **0.003** | 71.81% |

Chance level for 17 classes is 5.9%.

## What the trade-off chart shows

**Compression here is not a straightforward cost.** The most accurate model
in the study — pruned then quantised to int8, at 75.62% — is also 2.5x
smaller and 2.4x faster than the uncompressed baseline it improves upon.
It dominates the baseline on every axis simultaneously. There is no reason
to deploy the baseline.

Three models lie on the Pareto frontier and three are strictly dominated.
The frontier spans a real design space: at one end, 18.3 KB and 71.81% for
a device with severe memory limits; at the other, 32.1 KB and 75.62% where
2 percentage points matter more than 14 KB.

## What each technique actually did

**Quantisation was free.** Across every model, int8 conversion gave a
consistent ~2.5x size reduction and ~2.3x latency reduction with **zero
accuracy cost** (74.13% -> 74.13% on the baseline). On a battery-powered
wearable this matters beyond storage: many embedded processors lack a
floating-point unit entirely and must emulate float arithmetic in software.
Integer inference is what makes such a device viable at all.

The observed 2.5x falls short of the theoretical 4x (32-bit -> 8-bit)
because TFLite files also carry graph structure and per-tensor quantisation
parameters, which do not scale down. This fixed overhead is proportionally
larger for small models — a relevant constraint when targeting
microcontrollers.

**Pruning gave no size benefit — but improved accuracy.** This is the
study's most counterintuitive result. Driving 48.5% of weights to exactly
zero produced a TFLite file of 80.1 KB against the baseline's 79.9 KB:
no reduction whatsoever. Standard TFLite stores every weight explicitly,
and a zero occupies the same bytes as any other float. Sparsity pays off
only with a sparse storage format or sparsity-aware hardware, neither of
which plain TFLite provides.

An earlier, naive measurement appeared to show a 4.45x reduction from
pruning. That figure was an artefact: it compared a `.keras` file against
a `.h5` file, two formats carrying different metadata. Measuring both
through TFLite removed the confound and the effect vanished.

What pruning *did* deliver was regularisation: +1.33 percentage points
(74.13% -> 75.46%). Removing weak connections forced the network to rely
on strong ones, which generalised better.

**Distillation bought genuine compression.** A student with 3.3x fewer
parameters reached 72.87 +/- 1.04%, against 70.68 +/- 0.84% for an
identically-architected control trained on hard labels alone — a
distillation benefit of **+2.19 +/- 1.39 percentage points across five
random seeds**.

The control experiment was essential. Without it, the student's performance
could not be attributed to distillation rather than to the architecture
being adequate on its own. Furthermore, single-seed evaluation proved
actively misleading: two initial runs produced effects of **-2.99 and
+1.49 percentage points — of opposite sign**. Only repeated seeding
established that the effect is real and positive.

## Honest limitations

- **Single subject.** All results are within-subject (DB5 s1). sEMG models
  degrade sharply across users as electrode placement and anatomy vary;
  cross-subject robustness is the open problem in this field and is tested
  separately.
- **Rest windows excluded.** This gives a balanced 17-class problem
  consistent with standard DB5 protocol, but assumes gesture onset
  detection is handled upstream. A deployed system must discriminate
  rest from gesture.
- **Latency measured on a laptop CPU**, not on target hardware. The
  relative ordering should hold; absolute values will not.
- **Negligible 50 Hz mains contamination** was found in the spectrum,
  consistent with the Myo's wireless, battery-powered design. The notch
  filter was retained as standard practice but has minimal effect here.

## Design decisions driven by the edge constraint

- **200 ms windows.** Prosthetic and wearable control requires total
  latency under ~300 ms to feel responsive. The window is part of that
  budget.
- **Global average pooling instead of Flatten.** Flatten produces a large
  dense layer; GAP produces a small one. The architecture was chosen for
  the deployment target, not for maximum accuracy.
- **Split by repetition, not randomly.** The six repetitions of each
  gesture are near-identical; a random split places near-duplicates in
  both train and test, silently inflating results. Splitting by repetition
  (train 1/3/4/6, val 2, test 5) prevents this. Normalisation statistics
  were computed from the training set alone.
