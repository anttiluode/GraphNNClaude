# GraphNNClaude — does edge memory beat addressed slots when it has to hold more?

Follow-up to [GraphNN](https://github.com/anttiluode/GraphNN), the November 24, 2026 Neon control audit, and [GraphNNSol](https://github.com/anttiluode/GraphNNSol) Gate 1.

## Where things stand before this repo

- **Phase is not needed.** In Neon's controls (3 seeds, 896 bytes), a signed scalar edge tied complex phase on every mark test and had lower B error after chained rewrites (0.0182 vs 0.0245).
- **The rewrite geometry was already measured.** GraphNN PR #2 ([PR2_FINDINGS.md](https://github.com/anttiluode/GraphNN/blob/main/results/diagnostics/PR2_FINDINGS.md), October) found:
  - The actual rewrite moves B **13.6×** less than a random state change of the same size.
  - The local B Jacobian is **7.5×** less sensitive along the actual write direction than along a random one.
  - A sham rewrite of unchanged A changes 85.46% of bytes, against 85.88% for the real one.

  So the question "is the writer moving along B-preserving directions?" already has an answer: yes, locally, and B retention was a training target, not a discovery.
- **GraphNNSol Gate 1 covers the 2-image task.** Its arms are the valves (positive, signed, binary, phase), a node baseline and **two** fixed slots.

The 2-image task is at ceiling for every arm, so it cannot tell the arms apart. The one thing a distributed memory could do that an addressed memory cannot is **share capacity**: degrade gracefully when more items must fit into the same bytes. That is what this repo tests.

## The capacity gate (`capacity.py`)

K images are written one after another into **one** memory, item k at angle 90·k/(K−1). Then one random item per example has its 6×6 mark flipped and is rewritten. All K are read back by angle.

| Arm | Memory | Bytes | Bytes per item |
|---|---|---:|---:|
| positive | [0,1] edge gain, fixed degree | 896 | 896/K |
| signed | [−1,1] edge gain, fixed degree | 896 | 896/K |
| binary | exactly ±1 edges, packed bits | 112 | 112/K |
| phase | unit-modulus conjugate edge rotation | 896 | 896/K |
| slots | K disjoint channel groups, each item owns its own group; direct read, no transport | 896 | 896/K, fixed |

- **Shared components:** the writer, ping, decoder and loss are the same for every arm, and all arms have equal shared parameter counts (asserted in the self-test). The transports are Sol's Gate 1 code, generalised from 2 items to K.
- **Storage:** evaluation round-trips memory through uint8, or through packed bits for the binary arm, after every write.
- **Slot isolation:** the self-test asserts that writing one item into a slot leaves every other slot's bytes untouched.

**Measured, per K, on held-out procedural images:**

- `full`: after all K writes.
- `rewrite`: after flipping one item.
- `chain`: after rewriting all K items in turn.
- For each of these: per-item mark accuracy, all-K-marks-correct, and RGB MSE.
- `shuffled_floor`: the same reads from another example's memory. This is the floor, meaning what the decoder produces with the wrong memory.
- `untouched_drift`: how much the items that were not rewritten change when one item is rewritten.

### Interpretations fixed before any results

1. **Slots tie or beat the best edge arm at every K** (item mark within 1 percentage point, RGB MSE within 0.002). There is no distributed-memory advantage here. GraphNN's edge memory is then a slot memory with extra steps.
2. **Edge arms trail at K=2 but lead at K=8.** That is a crossover, and the graceful-degradation claim is supported on this task.
3. **Edge arms lead at every K.** That is suspicious until the slot read is given equal compute. The slot read is direct, with no 4-step transport, so FLOPs are not matched. Sol's Gate 1 notes the same caveat.
4. **An arm whose RGB MSE is within 10% of its own shuffled floor at some K** is not using its memory at that K, whatever its mark score says. Marks are 1 bit each, with 50% chance per item.
5. **Binary versus signed** is reported at 1/8 the storage. If binary is within 1 percentage point and 0.002 MSE of signed, sign-only edges are competitive at 1/8 the bytes. This overlaps Sol's binary arm and should agree with it at K=2.

## Run on Windows (RTX 3060)

```bat
git clone https://github.com/anttiluode/GraphNNClaude
cd GraphNNClaude
python capacity.py --self-test
python capacity.py --device cuda
```

The defaults are K = 2, 4, 8, seeds 41/97/127, 1500 steps, and all five arms. Each run writes `runs/capacity/summary.json` after every arm, with seed means at the end. For a quicker first look: `python capacity.py --device cuda --seeds 41 --steps 600`.

## Results

### CPU pilot (seed 41, 600 steps): not the preregistered run

One seed, 600 steps, on a 2-core CPU. Raw numbers are in `runs/pilot_cpu_seed41/summary.json`; this is a pilot, not the preregistered three-seed run. "Chain" means every item rewritten once, then all of them read. "Floor" is the reconstruction error when reading from another example's memory.

| K | Arm | Bytes per item | Chain: items right | Chain: all K right | Chain RGB MSE | Floor | Drift on untouched items |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | **slots** | 448 | 100% | 100% | **0.0112** | 0.095 | 0 |
| 2 | signed | 448 | 100% | 100% | 0.0180 | 0.091 | 0.00012 |
| 2 | phase | 448 | 100% | 100% | 0.0185 | 0.092 | 0.00036 |
| 2 | positive | 448 | 100% | 100% | 0.0202 | 0.089 | 0.00018 |
| 2 | binary | 56 | 100% | 100% | 0.0243 | 0.087 | 0.00046 |
| 4 | **slots** | 224 | 100% | 100% | **0.0152** | 0.092 | 0 |
| 4 | signed | 224 | 99.2% | 96.9% | 0.0413 | 0.079 | 0.0069 |
| 4 | phase | 224 | 97.7% | 90.6% | 0.0488 | 0.078 | 0.0070 |
| 4 | positive | 224 | 97.9% | 92.2% | 0.0462 | 0.078 | 0.0088 |
| 4 | binary | 28 | 95.3% | 81.2% | 0.0456 | 0.080 | 0.0086 |
| 8 | **slots** | 112 | 100% | 100% | **0.0216** | 0.091 | 0 |
| 8 | signed | 112 | 96.5% | 73.4% | 0.0554 | 0.071 | 0.0055 |
| 8 | phase | 112 | 85.4% | 21.9% | 0.0596 | 0.071 | 0.0044 |
| 8 | positive | 112 | 95.9% | 69.5% | 0.0596 | 0.075 | 0.0071 |
| 8 | binary | 14 | 80.2% | 14.1% | 0.0647 | 0.072 | 0.0033 |

**Reading, measured against the fixed interpretations:**

- **Interpretation 1 holds in the pilot at every K.** Slots get every mark right at every K and have the lowest image error.
- **The gap widens as the memory fills.**

  | K | Signed edges vs slots, chain image error |
  |---:|---:|
  | 2 | 1.6× |
  | 4 | 2.7× |
  | 8 | 2.6× |

  Slots with 112 bytes per image (K=8) reconstruct better than any edge arm with 448 bytes per image (K=2).
- **The edge arms do not use all their memory.** At K=8 their image error sits at 78–91% of their own wrong-memory floor, though none is within the 10% line. Binary after the chain sits at 90.5% of its floor, right at that line.
- **Among edge arms**, signed is best at K=4 and K=8. Phase and binary break down under repeated rewrites at K=8: only 22% and 14% of examples have all marks right.
- **These are fixed transports trained 600 steps on CPU.** The three-seed, 1500-step GPU run is the real test.

### Gate 2: is the loss in the reader or in the writer? (K=4)

The edge transport here is **linear in the ping** (asserted in the self-test), so the reader's whole view of the memory is one fixed linear operator applied to one probe. Two explanations are possible:

- **Reader bottleneck:** the bytes are all there, but one linear ping can't pull them out.
- **Writer interference:** the shared write itself smears items together.

Gate 2 separates them:

| Arm | Bytes stored | How it is read |
|---|---|---|
| direct | signed | Signed transport, **plus the raw bytes handed to the decoder**, as the slot reader gets them |
| signed_sweep | signed | Linear signed transport read at three ping pressures (0.5, 1, 2), each answer rescaled. On a linear medium the three are identical, so this is the null for the sweep. |
| elastic | signed | An edge conducts only once the pressure across it passes a threshold set by its byte: strong edges open early, weak edges late. This makes the medium nonlinear in the ping. |
| elastic_sweep | signed | Elastic edges read at the three pressures, like an oscillometric blood-pressure cuff, which sweeps cuff pressure to expose what a single reading can't |

**Fixed interpretations:**

- If `direct` closes most of the gap to slots, the reader is the bottleneck.
- If `direct` still loses clearly, the damage happens at write time, and no reader, cuff included, can fix it.
- `elastic_sweep` counts as a gain only if it beats both `elastic` and `signed_sweep` (the nonlinearity and the sweep each, alone).

**CPU result (seed 41, 600 steps, K=4).** Raw numbers are in `runs/gate2_cpu_seed41/summary.json`.

| Arm | Chain: items right | Chain: all 4 right | Chain RGB MSE | Floor | Drift on untouched items |
|---|---:|---:|---:|---:|---:|
| **slots** | 100% | 100% | **0.0152** | 0.092 | 0 |
| signed | 99.2% | 96.9% | 0.0413 | 0.079 | 0.0069 |
| direct (signed + raw bytes to the decoder) | 98.4% | 93.8% | 0.0416 | 0.076 | 0.0058 |
| elastic, single ping | 98.2% | 93.0% | 0.0448 | 0.080 | 0.0109 |
| signed_sweep (null) | 98.8% | 96.1% | 0.0456 | 0.077 | 0.0073 |
| elastic_sweep (the cuff) | 95.3% | 82.0% | 0.0462 | 0.077 | 0.0088 |

**Reading:**

- **The writer is the bottleneck, not the reader.** Handing the decoder every stored byte (`direct`, 0.0416) gives exactly what the transport alone gives (0.0413). The edge bytes simply do not contain the images that the slot bytes contain. The shared write mixes the four items together. Slots win because each item owns bytes that no other write touches; their drift is exactly zero.
- **The cuff read on elastic valves fails its preregistered test.** It does not beat elastic with a single ping (0.0462 vs 0.0448), nor the linear sweep null (0.0462 vs 0.0456). That matches the first point: once the bytes don't hold the information, no way of reading them can bring it back.
- **The pressure-dependent valve makes things slightly worse:** 0.0448 for elastic against 0.0413 for signed, and drift on untouched items rises to 0.0109.

So in this setup the question "is the loss in the reader or the writer" has its answer: the writer. Any further idea for this memory has to change how it is written, for example by giving the writer a way to keep items in separate bytes. Changing how it is read will not help.

Upstream `graphnn/` files are copied from GraphNN (MIT); the valve transports follow GraphNNSol Gate 1.
