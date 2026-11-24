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

_Pending._

Upstream `graphnn/` files are copied from GraphNN (MIT); the valve transports follow GraphNNSol Gate 1.
