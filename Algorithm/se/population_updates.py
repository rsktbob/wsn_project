"""Population replacement rules shared across SE flows and variants."""
import numpy as np


def searchers_from_goods(algorithm, goods, scores, selected):
    """Adopt the selected pool's best old good, strictly on improvement."""
    for searcher_id, region in enumerate(selected):
        good_id = int(np.argmax(scores[region]))
        score = float(scores[region, good_id])
        if score > algorithm.searcher_fitness[searcher_id]:
            algorithm.searchers[searcher_id] = goods[region][good_id].copy()
            algorithm.searcher_fitness[searcher_id] = score


def searchers_from_children(algorithm, children, scores):
    """Only a searcher's own best child may replace that searcher."""
    for searcher_id in range(algorithm.n):
        good_id = int(np.argmax(scores[searcher_id]))
        score = float(scores[searcher_id, good_id])
        if score > algorithm.searcher_fitness[searcher_id]:
            algorithm.searchers[searcher_id] = children[searcher_id][good_id].copy()
            algorithm.searcher_fitness[searcher_id] = score


def goods_from_children(goods, goods_fitness, children, scores, visitors,
                        *, tolerance=None, accept=None):
    """Replace each slot from its visitors, preserving first-winner ties.

    tolerance=None retains the original unconditional goods replacement.
    A numeric tolerance enables strict improvement over the old good.
    """
    if not len(visitors):
        return
    for good_id in range(len(goods)):
        winner = int(max(visitors, key=lambda i: scores[int(i), good_id]))
        score = float(scores[winner, good_id])
        if tolerance is not None and score <= goods_fitness[good_id] + tolerance:
            continue
        if accept is not None and not accept(goods_fitness[good_id], score):
            continue
        goods[good_id] = children[winner][good_id].copy()
        goods_fitness[good_id] = score
