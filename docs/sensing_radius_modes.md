# Sensing radius modes

`Problem` owns the physical sensing-range model.  Decoded states keep integer
option indexes in `state.sch`; `Problem.resolve_state_radius(state)` translates
them into the physical `state.radius` used by coverage, sensing energy, fitness,
life checks, routing capacity, metrics, and drawings.

## Modes

- `discrete` (default): every sensor uses the configured global levels.  The
  default values are `0, 3, 6, 9, 12, 15`.
- `bucketed`: the discrete levels define intervals `(0, 3]`, `(3, 6]`, and
  so on.  A sensor receives one option for an interval only when that interval
  contains newly reachable targets; the option radius is the farthest target
  distance in the interval.  Empty intervals create no option, so each sensor
  can have fewer than six actions.
- `exact`: sensor `i` receives the sorted unique distances from that sensor to
  all targets within `maximum_sensing_radius`, plus zero for the off state.
  These target-distance event points retain every energy-minimal radius under
  deterministic disk coverage and the quadratic sensing-energy model.

Example:

```python
problem = Problem(
    B=100,
    S=100,
    T=100,
    F=10,
    FILE="A2",
    sensing_mode="exact",
)
```

Experiments expose the same choice:

```powershell
python experiment_algorithms.py `
  --algorithm gi_gomea_target `
  --mode lifetime `
  --sensing-mode exact
```

For the compact interval-based mode:

```powershell
python experiment_algorithms.py `
  --algorithm sa_sets `
  --mode lifetime `
  --sensing-mode bucketed
```

For example, target distances `2.6, 8.2, 8.3, 14.7` produce bucketed options
`[0, 2.6, 8.3, 14.7]`.  The empty `(3, 6]` and `(9, 12]` intervals do not
consume action indexes.  The corresponding exact options are
`[0, 2.6, 8.2, 8.3, 14.7]`.

In bucketed mode, `sensing_levels` are the interval boundaries and their last
value must equal `maximum_sensing_radius`.  With the defaults, every sensor has
at most six actions including the off action, but it may have fewer.

## Decoded-state contract

All supported states should provide:

- `sch`: integer sensing-option index per sensor;
- `radius`: physical sensing radius per sensor, synchronized by `Problem`;
- `rou`: next-hop device id per sensor;
- `use`: generated and forwarded traffic per sensor.

Coding classes may use different chromosomes and decoders, but all decoders
return the same `State`. `CriticalPatternCoding` owns the historical
pattern/override representation and follows the same Problem-owned sensing
mode as every other coding; it has no special continuous-cost path.

## Compatibility

`LEVEL_RANGE` and `COST_RANGE` remain as legacy discrete configuration fields;
`bucketed` also uses `LEVEL_RANGE` as its interval boundaries.
New physical calculations use `radius_table` and `SENSING_COST`, both indexed by
`[sensor_id, option_id]`.  Code that needs a physical value should call:

```python
problem.sensing_radius(sensor_id, option_id)
problem.sensing_cost(sensor_id, option_id)
problem.resolve_radius(schedule)
```
