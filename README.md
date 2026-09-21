# flydoom

A real *Drosophila melanogaster* connectome plays DOOM.

No neural network trained on anything. The synaptic weights are measured from
electron microscopy of an actual fly brain, and the signs come from measured
neurotransmitter predictions. It plays badly, and that is the finding.

## Result (20 seeds, ViZDoom `defend_the_center`, kills per episode)

| player | kills/ep |
|---|---|
| connectome, real neurotransmitters | **1.15** |
| random actions | 1.45 |
| idle (do nothing) | 0.00 |
| connectome, randomised synapse signs | 0.60 |

Mann-Whitney vs random: p = 0.19. The connectome is reliably better than doing
nothing and statistically indistinguishable from a coin flip.

Using the *real* neurotransmitter labels instead of randomised signs roughly
doubled performance (0.60 to 1.15). Sign matters more than people assume.

## Data

- **Connectome**: Janelia hemibrain v1.2 traced adjacencies, 21,739 neurons /
  3.55M synapses. <https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz>
- **Neurotransmitters**: Eckstein & Bates et al. 2024, Zenodo record 10593546,
  `supplemental_data_3.csv`. 99.6% bodyId coverage, 36% of neurons inhibitory.
  ACh = +1, GABA = -1, glutamate = -1 (GluCl is inhibitory in flies), amines = +0.25.

## Architecture

Leaky integrate-and-fire over the real adjacency matrix. Hemibrain is a single
hemisphere, so it is mirrored into two; without that there is no left/right
steering at all. Reciprocal inhibition between the DNa01/DNa02 steering pools.
Operating point: 3.6% population firing.

**In** — LC4/LPLC2/LC6/LPLC1 (looming), LC11/12/18 (small object), LC10
(tracking), LC9/16/17 (wide-field), driven by 8 retinotopic columns of the
framebuffer.

**Out** — DNa01/DNa02 graded membrane potential to yaw; DNp02/04/06/09 escape
descending neurons to ATTACK; MDN to backward (unused here).

## What is not biology

Be suspicious of anyone who skips this section.

- **No optic lobe.** Hemibrain does not contain one, so the fly's entire early
  visual system is replaced by 8 hand-written feature channels.
- **Receptive fields are fake.** LC neurons are assigned to columns by array
  index, not by anatomy.
- **Adaptation and gain control are added.** A raw wiring diagram is bistable:
  it is either silent (0% firing, the fly stands still and dies) or seizing
  (45% firing). Spike-frequency adaptation plus homeostatic gain control were
  needed to reach a plausible sparse regime. The connectome does not record this.
- **No learning anywhere.**

A static wiring diagram is not a brain. Most of the competence here comes from
the scaffolding, not the connectome.

## Negative results kept on purpose

- Giving each hemisphere its own visual hemifield (anatomically more correct)
  made performance *worse* (0.90), even after fixing steering polarity.
- A 5-seed sweep found a config scoring 1.80. On 20 seeds it regressed to 1.15.
  That 1.80 was noise. It is the number a careless writeup would headline.

## Run it

```bash
pip install vizdoom scipy pandas numpy
python flydoom_v2.py
```

Downloads both datasets on first run (~50 MB) and reproduces the 20-seed benchmark.

## Next

- Swap hemibrain for FlyWire: full brain, both hemispheres, real optic lobe,
  real neuron coordinates. Fixes the two biggest cheats. Needs a Codex token.
- Evolve only the LC input weights, connectome frozen. Would likely beat random,
  but then the connectome is no longer doing the work alone.

MIT.
