# PSO with SensorEncoding

`PSO` searches the same two genes per sensor as `GA`: a sensing option and a routing priority (0–9). Each particle has continuous positions in `[0, 1]`; a position is mapped to an integer using that gene's own domain before the shared `SensorEncoding.decode()` and fitness evaluation. Positions and velocities remain continuous for the PSO update. The initial swarm is uniform, and each iteration updates personal and global bests before applying inertia, cognitive, and social velocity terms. A velocity cap of `0.2` and position clipping keep the search bounded.

The experiment registry name is `pso`. For example:

```sh
python experiment_algorithms.py --algorithm pso --runs 1
```

The preset uses 30 particles, inertia `0.7`, cognitive/social coefficients `1.5`, and velocity cap `0.2`. These are exploratory defaults, not tuned WSN parameters. The evaluation budget counts each decoded chromosome exactly once, including a partial final swarm iteration. `history` records the best score after each evaluation.
