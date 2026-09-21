# flydoom

A real *Drosophila melanogaster* connectome plays DOOM.

No neural network trained on anything. The synaptic weights are measured from
electron microscopy of an actual fly brain, and the signs come from measured
neurotransmitter predictions. It plays badly, and that is the finding.

## Result (n=20 seeds, ViZDoom `defend_the_center`, kills per episode)

| player | kills/ep | vs random |
|---|---|---|
| **FlyWire full brain** (v3) | **1.25** | p = 0.47 |
| hemibrain, real neurotransmitters (v2) | 1.15 | p = 0.19 |
| random actions | 1.45 | baseline |
| idle (do nothing) | 0.00 | floor |
| hemibrain, randomised synapse signs (v1) | 0.60 | superseded |

The full brain modestly beat the single-hemisphere hemibrain (1.25 vs 1.15),
but the two are not distinguishable at n=20 (p = 0.45), and neither beats
random action selection. More anatomy did not buy a statistically significant
win. That's a real result, not a bug.

## Data

**v3 (current)** — FlyWire FAFB, full brain, both hemispheres, 139,255 neurons,
15,091,983 synapses, per-synapse neurotransmitter probability scores.
[SLOP011/flywire-fafb-connectome](https://huggingface.co/datasets/SLOP011/flywire-fafb-connectome)
on Hugging Face (public, no auth). Underlying papers: Dorkenwald et al. 2023
(wiring), Schlegel et al. 2023 (cell typing), Eckstein & Bates 2024 (transmitters).

**v2 (superseded)** — Janelia hemibrain v1.2, single hemisphere, 21,739 neurons,
mirrored in software to get bilateral steering.
[storage.googleapis.com/hemibrain](https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz)
+ [Zenodo 10593546](https://zenodo.org/records/10593546) for neurotransmitters.

## Architecture

Leaky integrate-and-fire over the real signed adjacency matrix. Synapse sign in
v3 is continuous: `(ach + 0.25*(da+ser+oct)) - (gaba+glut)` per synapse, not a
single discrete label per neuron. Spike-frequency adaptation + homeostatic gain
control (a raw wiring diagram is bistable: silent or seizing without them).
v3 settles at ~1.0% population firing vs v2's 3.6% — the full brain's extra
inhibitory circuitry changes network gain, and the constants weren't re-tuned
for it, which is itself an honest limitation.

**In** — LC4/LPLC2/LC6/LPLC1 (looming), LC11/12/18 (small object), LC10
(tracking), LC9/16/17 (wide-field), matched by `hemibrain_type` cross-reference
so the same neuron classes are used in both v2 and v3. Driven by 8 columns of
the Doom framebuffer.

**Out** — DNa01/DNa02 by real left/right side → yaw. DNp02/04/06/09 escape
descending neurons → ATTACK. MDN → backward (unused in this scenario).

## What is not biology

- **LC receptive fields are still fake.** Assigned to 8 columns by array index.
  FlyWire's actual optic lobe (lamina, medulla, lobula) was not traversed —
  only the LC/LPLC relay neurons, matched by name. Real retinotopy would
  require walking the optic lobe columns, which this does not do.
- **Scaffolding constants weren't re-tuned for the bigger brain.** Same leak,
  threshold, and gain-control target as v2, just applied to 6.4x more neurons.
- **No learning anywhere**, in either version.

A static wiring diagram is not a brain. Most of the competence here — such as
it is — comes from the LIF scaffolding, not the connectome.

## Negative results kept on purpose

- v2: giving each hemisphere its own visual hemifield made performance worse
  (0.90), even after fixing steering polarity.
- v2: a 5-seed sweep found a config scoring 1.80. On 20 seeds it regressed to
  1.15. That 1.80 was noise — the number a careless writeup would headline.
- v3: five seeds in a row scored identically 1 kill early in testing, which
  looked like a stuck network. It wasn't — six more seeds immediately broke the
  pattern (0–3 range). Worth checking before trusting a suspiciously flat run.

## Run it

```bash
pip install vizdoom scipy pandas pyarrow numpy caveclient
python flydoom_v3.py     # full FlyWire brain (current)
python flydoom_v2.py     # hemibrain, single hemisphere (superseded, kept for comparison)
```

flydoom_v3.py downloads the ~475 MB FlyWire connection table on first run (public,
no auth). flydoom_v2.py downloads hemibrain + Zenodo NT predictions (~50 MB).

## Next

- Re-tune the LIF constants (leak, threshold, gain target) specifically for the
  full brain instead of reusing hemibrain's — the firing-rate mismatch (1.0% vs
  target 3.0%) suggests there's headroom here.
- Walk the actual optic lobe for real retinotopic receptive fields.
- Evolve only the LC input weights, connectome frozen. Would likely beat
  random, but then the connectome is no longer doing the work alone.

MIT.
