# RL-SETSv6.1

RL-SETSv6.1 keeps the SI-SETSv2 role-separated elitist market update while
learning crossover and mutation strength independently.

## Contract

- Observation: 22 normalized candidate, region, search-progress and energy fields.
- Action: 3 crossover modes times 3 mutation modes (9 actions).
- Action 4: original SA-SETS crossover/mutation rules for each role-separated
  offspring.
- Reward: feasibility first; accepted updates receive 70% relative fitness and
  30% predicted-lifetime credit. Rejected ordinary damage is multiplied by
  0.1, while destroyed feasibility keeps its full negative credit.
- Replay: 200,000 transitions by default, capped at 1,000 transitions per map
  and sampled uniformly across represented maps.
- Checkpoint: network, target network, Adam state, replay and learner RNG.

## Short smoke training

```powershell
python train_rl_sets.py --algorithm rl_setsv61 `
  --map-limit 2 --episodes 2 --evaluate 40 --max-rounds 1 `
  --max-lifetime 1 --validation-every 0 `
  --rl-model Models/rl_setsv61_smoke.npz
```

## Full 200-map training

The default v6.1 training cohort uses all 200 training maps, one pass, an
epsilon floor of 0.10, and validates once at episode 200 against the manifest's
40 held-out maps. Validation is expensive; pass a smaller validation manifest
when validating more frequently.

```powershell
python train_rl_sets.py --algorithm rl_setsv61 `
  --rl-model Models/rl_setsv61_200maps_1pass.npz
```

## Frozen experiments

Trained policy:

```powershell
python experiment_algorithms.py --algorithm rl_setsv61 `
  --rl-model Models/rl_setsv61_200maps_1pass.best.npz
```

Uniform random over all nine actions:

```powershell
python experiment_algorithms.py --algorithm rl_setsv61
```

The fixed-action baselines (action 4 = the original SA-SETS perturbation,
action 0 = local/local) are no longer reachable from the experiment CLI; the
`--rl-fixed-action` flag was removed. `RL_SETSv61` still accepts the argument
directly, so a baseline run means constructing it yourself:

```python
from Algorithm.se.RL_SETSv61 import RL_SETSv61

algorithm = RL_SETSv61(problem, n=8, h=4, w=2, mu=0.4, fixed_action=4, seed=7)
```

To bring the flag back, add `fixed_action` to the `rl_setsv61` preset in
`experiments/presets.py` and a `--rl-fixed-action` argument in
`experiments/cli.py`.

Do not load an RL-SETSv6 checkpoint into v6.1. The observation and action
schemas are intentionally incompatible.
