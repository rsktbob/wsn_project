# SensorEncoding GA family

`GA` directly inherits `Algorithm` and holds the generational loop: initialize and evaluate a population, select parents, crossover, mutate, and evaluate the offspring under the requested fitness-evaluation budget. `GA` supplies the SensorEncoding chromosome and fixed-rate operators. `ElitistGA` keeps evaluated elites without re-evaluation.

| CLI name | Method | Default |
| --- | --- | --- |
| `ga` | Tournament selection, fixed two-point crossover and 2023-thesis mutation (each chromosome with probability mu redraws one random gene) | n=30, cu=0.8, mu=0.1 |
| `ga_elitist` | As GA, plus two cached elites | n=30, elite_count=2 |
| `ga_adaptive` | Srinivas–Patnaik (1994) fitness-dependent crossover and mutation probabilities | n=30, k1=k3=1, k2=k4=0.5, baseline mutation=0.005 |

Every sensor has `[sensing option, routing priority]` integer genes. Sensing domains come from `radius_option_counts`; priorities are 0–9. All three methods use the shared `SensorEncoding.decode()` and fitness service.

## Paper-based adaptive rates

The 1994 paper is [Srinivas and Patnaik, *Adaptive probabilities of crossover and mutation in genetic algorithms*](https://doi.org/10.1109/21.286385), equations (5)–(8). In each generation, let `f_avg` and `f_max` be the population's mean and maximum fitness, `f_pair` the better fitness of a selected parent pair, and `f` the selected individual's fitness:

- If `f_pair < f_avg`, `p_c = k3`; otherwise `p_c = k1 (f_max - f_pair)/(f_max - f_avg)`.
- If `f < f_avg`, `p_m = k4`; otherwise `p_m = k2 (f_max - f)/(f_max - f_avg)`.
- The paper also specifies a baseline mutation rate of 0.005. This adapter applies it as a floor: `max(0.005, p_m)`.

If `f_max == f_avg`, the published fractions are undefined. This implementation uses the low-fitness branch (`k3`, `k4`) to keep exploring. Parent selection is fitness-proportional; when WSN fitness has nonpositive values, weights are shifted to nonnegative values (or made uniform if all weights are zero). The original paper uses a binary alphabet. This WSN adaptation retains the project's coverage-seeded initialization, sensor-aligned two-point crossover, and domain-valid integer mutation. The equations and published coefficients are retained; the encoding, fitness and these explicit edge-case choices are different from the original experiments. Mutation probabilities are based on selected parents' evaluated fitness before crossover, so no additional fitness calls occur between crossover and mutation.

Run from `codes/2027WSN`:

```sh
python experiment_algorithms.py --algorithm ga ga_elitist ga_adaptive --mode single --runs 1 --evaluate 1000
python tests/test_ga_variants.py
```
