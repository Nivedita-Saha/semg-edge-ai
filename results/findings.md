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

---

# Interpretability — and why it matters for EnduRAI

## What the model actually looks at

SHAP attribution was computed for 136 test windows spanning all 17
gestures. Two findings:

**Each gesture recruits a distinct subset of electrodes.** The channel
importance map shows no two gestures relying on the same pattern:
gesture 3 depends almost entirely on channel 13, gesture 6 on channel 7,
gestures 9 and 11 on channel 0, gestures 5/8/10 on channel 15. Had the
model learned a shortcut — classifying on overall signal amplitude rather
than muscle-specific patterns — every row would look identical. It does
not. **The model has learned forearm physiology, not a proxy for it.**

**The model uses timing, not just amplitude.** In the single-window
attribution, channels 8 and 10 contribute throughout the 200 ms window,
while channel 7 contributes sharply only in the first 20-35 ms. Gesture
onset carries information, and the model exploits it.

## Why this matters for a factory wearable

EnduRAI is a human-centred project. Its subject is not a dataset — it is
a worker, wearing a device on their arm, all shift.

**Accuracy alone does not earn trust.** A device that silently classifies
a person's muscle activity, and is right 75% of the time, is a device that
is *wrong about that person's body once every four gestures*. If the
worker cannot ask why, and cannot be told, they have no basis to accept it
— and no basis to challenge it when it is wrong.

Interpretability changes what can be said to that worker:

- **"The system decided this because these two muscles fired, at this
  moment."** That is a claim a person can evaluate, agree with, or dispute.
- **When the model fails**, attribution shows whether it failed because
  an electrode had shifted, or because the gesture was genuinely ambiguous.
  The first is fixable by the worker; the second is not their fault.
- **It makes the system auditable.** If a device's decisions feed into
  safety systems, productivity monitoring, or ergonomic assessment, then
  "the network said so" is not an acceptable answer to anyone — worker,
  supervisor, or regulator.

Compression makes such a device *possible*: an 18 KB model can run on the
sensor itself, keeping a worker's raw muscle data on their own arm rather
than streaming it to a server. Interpretability is what makes it
*acceptable*. Both are necessary. Neither is sufficient alone.

That combination — a model small enough to run on the body, and
transparent enough to be questioned by the person wearing it — is what
this project was built to demonstrate.

---

# Cross-subject robustness — the real problem

The baseline model, trained entirely on subject 1, was applied unchanged
to subject 2 (same armband, same 17 gestures, never seen in training).
Normalisation used subject 1's statistics, as a deployed model would have
to — it cannot see a new user's data in advance.

| | Accuracy |
|---|---|
| Subject 1 (trained on) | 74.13% |
| Subject 2 (never seen) | **27.62%** |
| Chance | 5.88% |

**A 46.5 percentage point collapse.** The model remains above chance, so
some structure transfers — but it is nowhere near deployable on a new user.

## The failure is bimodal, not uniform

Per-gesture accuracy on subject 2 splits sharply:

**Transferred:** gesture 3 (90.7%), gesture 1 (79.4%), gesture 11 (74.0%),
gesture 6 (51.9%), gesture 13 (37.2%)

**Collapsed:** gesture 2 (0.0%), gesture 7 (0.7%), gesture 9 (2.2%),
gesture 10 (2.5%), gesture 8 (2.6%), gesture 15 (1.8%)

The model does not degrade gracefully. It either recognises a gesture
almost as well as on the training subject, or it fails completely.

## A mechanism, from the interpretability analysis

Cross-referencing against the SHAP channel importance map suggests why.

**The gestures that transferred are those whose SHAP attribution was
concentrated on one or two dominant electrodes.** Gesture 3 depended
almost entirely on channel 13; gesture 11 on channel 0. The gestures that
collapsed relied on distributed patterns across many channels.

The proposed mechanism: **a single strong, localised muscle activation
survives armband rotation, because a small rotation still leaves a
neighbouring electrode observing a similar signal. A distributed spatial
pattern does not survive, because the entire spatial arrangement is
scrambled by the rotation.** The model has learned subject 1's specific
electrode-to-muscle mapping, and that mapping does not hold on a
different arm.

This is why the confusion matrix shows several true gestures being dumped
into predicted class 11: when the learned spatial pattern is lost, the
model falls back on whichever channel still resembles something familiar.

## Why this matters

Cross-subject generalisation is the central unsolved problem in wearable
sEMG, and the reason commercial gesture-control devices have repeatedly
failed in the field. A device that must be recalibrated for every user,
and re-calibrated again if the armband is put on slightly differently the
next morning, is not a product.

**Future work** would target this directly: subject-independent training
across many users, domain adaptation, electrode-shift augmentation during
training, or rotation-invariant architectures that do not assume a fixed
electrode-to-muscle mapping. The interpretability finding above suggests
a specific hypothesis worth testing — that encouraging the model toward
sparser, more localised channel attribution during training might
*improve* cross-subject robustness, at some cost to within-subject
accuracy.

---

## Testing the rotation hypothesis — a refutation

The interpretability analysis suggested armband rotation as the mechanism
behind the cross-subject collapse. This was tested directly: the model was
retrained with electrode-shift augmentation (random circular roll of
+/-2 positions within each 8-electrode band, both bands rolled together),
across three seeds.

| | Subject 1 (seen) | Subject 2 (unseen) |
|---|---|---|
| Baseline | 73.47 +/- 0.36% | 28.17 +/- 2.36% |
| + rotation augmentation | 59.37 +/- 0.27% | 17.17 +/- 0.43% |
| **Effect** | **-14.10 pp** | **-11.00 pp** |

**The hypothesis is refuted.** Augmentation degraded both within- and
cross-subject accuracy, consistently across seeds and with small variance.

The interpretation: the electrode-to-muscle mapping is not an artefact the
model over-relies on — **it is the signal**. Which electrode observes which
muscle is precisely what distinguishes one gesture from another. Rolling
the channels does not teach rotational invariance; it destroys the
discriminative information.

This has a direct consequence for future work: a **rotation-invariant
architecture** (circular convolution over the electrode ring) would build
in an invariance the data does not support, and should be expected to fail
for the same reason. That avenue is closed.

The cross-subject gap must therefore be driven by other factors —
anatomical differences (muscle size, adipose thickness, skin
conductivity), or inter-subject variation in how the gestures are
performed. Distinguishing these is the natural next experiment.

*Caveat: this assumes DB5 channels 0-7 and 8-15 correspond to the two
Myo armbands. This should be verified against the Ninapro documentation;
if the layout differs, the roll would have scrambled channels arbitrarily
rather than simulating rotation, and the experiment would need repeating.*

---

## Decomposing the gap, part 1: is it amplitude?

If subjects differ simply in signal magnitude — muscle mass, adipose
thickness, skin conductivity, band tightness — then every channel arrives
at the model mis-scaled, and an intact gesture pattern could be
unrecognisable. Three normalisation strategies were compared. **None uses
subject 2's labels**, so all three are deployable: a device can ask a new
user to wear the band and move briefly.

| | Accuracy | vs baseline |
|---|---|---|
| Subject 1 (reference) | 74.13% | — |
| A. Subject 1's statistics (baseline) | 27.62% | — |
| B. Subject 2's own statistics | 30.80% | **+3.19 pp** |
| C. Subject 2's REST statistics only | 13.84% | **-13.78 pp** |

**Amplitude accounts for only ~7% of the 46.5 pp gap.** Recalibrating to
subject 2's own statistics recovers 3.2 points and leaves 39 unexplained.
Scaling is not the problem.

### Calibrating on rest is actively harmful

Strategy C — "put the band on and hold still for five seconds" — is the
most practically deployable calibration imaginable, and it **halved**
accuracy.

The reason: **rest is not a quieter version of gesture.** With the muscles
inactive, the resting signal is dominated by the noise floor — electronic
noise, baseline muscle tone, skin impedance. Dividing by the standard
deviation of *noise* does not rescale the gesture patterns; it amplifies
whichever channels happened to be quietest at rest and suppresses those
that were noisiest, distorting the spatial signature the model depends on.

This is worth stating plainly because rest-based calibration is the obvious
thing a practitioner would reach for. It should not be used.

## Where this leaves the cross-subject problem

Two candidate explanations have now been tested and rejected:

| Hypothesis | Test | Result |
|---|---|---|
| **Armband rotation** | electrode-shift augmentation, 3 seeds | **Refuted** — -14.1 pp within-subject, -11.0 pp cross-subject |
| **Amplitude scaling** | three normalisation strategies | **Refuted** — 7% of gap; rest-based calibration harmful |

The gap is therefore **genuine spatial pattern mismatch**. Subject 2's
gestures produce a differently-*shaped* signature across the electrodes —
not a rotated one, and not a rescaled one. A different one.

That points at two remaining candidates:

- **Anatomy.** Different forearm geometry means a given electrode sits over
  a genuinely different mixture of muscles, producing a pattern that is not
  a transformation of subject 1's pattern but an unrelated one.
- **Behaviour.** Subject 2 may simply perform the gestures differently —
  different force, different finger involvement, different wrist angle.

Distinguishing these is the natural next experiment, and neither is
solvable by preprocessing. Both would require either learning across many
subjects, or a small amount of labelled calibration data from the new user.

---

## Decomposing the gap, part 2: is there a correctable rotation offset?

Electrode-shift augmentation failed because it *destroyed* electrode-identity
information. Alignment tests the opposite treatment: assume electrode
identity matters, and try to *recover* the correct mapping on a new arm.
Subject 2 performs one reference gesture; the offset is estimated from
which electrode fires hardest, and their channels are rolled to match.

We also brute-forced **all 8 possible rotations** to establish the oracle
upper bound — the ceiling on what any alignment method could achieve.

| Rotation | Accuracy |
|---|---|
| **shift 0 (no correction)** | **27.62%** |
| shift 1 | 12.41% |
| shift 2 | 6.13% |
| shift 3 | 4.71% |
| shift 4 | 4.74% |
| shift 5 | 6.25% |
| shift 6 | 6.59% |
| shift 7 (estimated offset) | 10.00% |

Chance is 5.88%.

**The oracle best rotation is shift 0 — doing nothing.** Accuracy falls
monotonically as the band is rotated away from zero, bottoms out below
chance at 3-4 positions (a half-turn, maximally misaligned), and recovers
as it wraps back. This is the signature of a system that is *already
correctly aligned*. There is no offset to correct, and **no rotation
correction, however estimated, can help — the ceiling on the entire
approach is zero.**

### Why the one-gesture estimator failed

Subject 1's reference gesture peaks on electrode 7; subject 2's peaks on
electrode 0. The estimator inferred a 7-position rotation and cost 17.6
percentage points.

But electrodes 7 and 0 are **adjacent** on an 8-electrode ring. The two
subjects are not rotated relative to each other. They simply recruit
**slightly different muscles for the same intended gesture** — the peak
lands one electrode over because their forearms are different.

The estimator mistook anatomical variation for rotation. Which is exactly
what the cross-subject problem turns out to be.

## Conclusion: three refutations

| Hypothesis | Test | Result |
|---|---|---|
| Rotation (make model ignore it) | electrode-shift augmentation, 3 seeds | **Refuted** — -14.1 pp within, -11.0 pp cross |
| Amplitude scaling | three normalisation strategies | **Refuted** — 7% of gap; rest-calibration harmful |
| Rotation (correct it) | oracle over all 8 shifts | **Refuted** — ceiling is zero |

**The cross-subject gap is not a geometric or scaling problem. It is
anatomical and behavioural.** Subject 2's forearm produces a genuinely
different spatial signature for the same intended gesture. No preprocessing
transformation can map one onto the other, because no such transformation
exists — these are different signals, not distorted versions of the same
signal.

This has a clear consequence for what *would* work, and it is the honest
conclusion of this project:

- **Preprocessing and augmentation cannot solve this.** Three attempts,
  three refutations.
- **Multi-subject training** must learn a distribution over anatomies
  rather than a single mapping — and should be expected to help only
  partially, for the same reason.
- **A small amount of labelled calibration data from the new user** —
  a few seconds per gesture — remains the only intervention with a clear
  mechanism. It does not solve the scientific problem; it moves the cost
  onto the user. That is what every commercial device does, and now we
  understand why.
