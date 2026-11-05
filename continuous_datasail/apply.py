"""Applying ranked operations, and choosing how many of them to apply.

Once the operations are ordered, the interpolation itself is simply "apply the
first k". The only question left is which k corresponds to a requested
percentage, and there are two answers: search along the leakage axis, or divide
the list of operations proportionally.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .graph import Operation
from .leakage import leakage


def apply_cycle(split: Dict[str, str], operation, end_split: Dict[str, str]):
    """Apply one operation to a split and return the new split.

    A cycle is rotated: every member takes the fold of the next member, so each
    fold in the cycle gives up one entity and receives one, and all fold sizes
    are preserved. A leftover is a single entity, which is simply sent to the
    fold it has in ``end_split``; that one does change fold sizes.

    Parameters
    ----------
    split : dict
        Entity name -> fold. Not modified.
    operation : Operation
        Entity names. Length > 1 is a cycle, length 1 a leftover.
    end_split : dict
        Entity name -> fold at the end of the interpolation. Only consulted for
        leftovers.

    Returns
    -------
    dict
        A new split; the input is left alone.
    """
    new_split = dict(split)
    n = len(operation)
    if n == 1:
        name = operation[0]
        new_split[name] = end_split[name]
        return new_split
    for i, name in enumerate(operation):
        next_name = operation[(i + 1) % n]
        new_split[name] = split[next_name]
    return new_split


def apply_k_cycles(
        start_split: Dict[str, str],
        ranked_operations: List[Tuple[Operation, float]],
        end_split: Dict[str, str],
        k: int,):
    """Apply the first ``k`` ranked operations to the start split.

    Parameters
    ----------
    start_split : dict
        Entity name -> fold at the start. Not modified.
    ranked_operations : list of (Operation, float)
        Output of a ranking function.
    end_split : dict
        Entity name -> fold at the end, consulted for leftovers.
    k : int
        How many operations to apply. ``k == 0`` reproduces the start split and
        ``k == len(ranked_operations)`` reproduces the end split.

    Returns
    -------
    dict
        The interpolated split.
    """
    split = dict(start_split)
    for operation, _ in ranked_operations[:k]:
        split = apply_cycle(split, operation, end_split)
    return split


@dataclass
class SplitResult:
    """One interpolated split and the numbers describing where it landed.

    Attributes
    ----------
    split : dict
        Entity name -> fold.
    k : int
        Number of operations applied.
    n_operations : int
        Length of the whole operation list, so ``k / n_operations`` is the
        fraction of the decomposition consumed.
    leakage_start, leakage_end : float
        Leakage of the two endpoints.
    leakage_achieved : float
        Leakage of this split.
    percentage_requested : float
        What the caller asked for, on a 0-100 scale.
    percentage_achieved : float
        Where the split actually landed, as a fraction of the way from
        ``leakage_start`` to ``leakage_end``. Note the scales differ: requested
        is 0-100, achieved is 0-1.
    """
    split: Dict[str, str]
    k: int
    n_operations: int
    leakage_start: float
    leakage_end: float
    leakage_achieved: float
    percentage_requested: float
    percentage_achieved: float


def generate_split(start_split, end_split, ranked_operations, names, metric, percentage, weights=None):
    """Pick k by searching along the leakage axis for the requested percentage.

    The target is ``L_start + percentage/100 * (L_end - L_start)``, and a binary
    search looks for the k whose split comes closest to it.

    The search assumes leakage rises or falls monotonically with k. Under
    leakage ranking it does not: the curve dips before climbing, because the
    deltas are scored against the start split and never rescored. Near the dip
    the search can therefore return a k some way from the best one. Use
    ``k_method="linear"`` when that matters more than hitting the target
    leakage exactly.

    Parameters
    ----------
    start_split, end_split : dict
        Entity name -> fold at each endpoint.
    ranked_operations : list of (Operation, float)
        Output of a ranking function.
    names : list of str
        Entities, in the same order as the rows of ``metric``.
    metric : numpy.ndarray
        Square similarity matrix.
    percentage : float
        Target, on a **0-100** scale. Passing 0.5 means half of one percent,
        not halfway.
    weights : dict, optional
        Entity name -> weight.

    Returns
    -------
    SplitResult

    Raises
    ------
    ValueError
        When ``percentage`` is outside [0, 100].
    """
    if not (0 <= percentage <= 100):
        raise ValueError("percentage must be between 0 and 100")
    L_start = leakage(start_split, names, metric, weights)
    L_end = leakage(end_split, names, metric, weights)
    target = L_start + percentage / 100 * (L_end - L_start)
    n_ops = len(ranked_operations)
    increasing = L_end >= L_start

    lo, hi = 0, n_ops
    best_k, best_split, best_L = 0, start_split, L_start
    while lo <= hi:
        mid = (lo + hi) // 2
        split_mid = apply_k_cycles(start_split, ranked_operations, end_split, mid)
        L_mid = leakage(split_mid, names, metric, weights)
        # Keep the closest split seen, not just the one the search ends on.
        if abs(L_mid - target) < abs(best_L - target):
            best_k, best_split, best_L = mid, split_mid, L_mid
        if (L_mid < target) == increasing:
            lo = mid + 1
        else:
            hi = mid - 1

    achieved = 0.0 if L_end == L_start else (best_L - L_start) / (L_end - L_start)
    return SplitResult(split=best_split, k=best_k, n_operations=n_ops,
                       leakage_start=L_start, leakage_end=L_end, leakage_achieved=best_L,
                       percentage_requested=percentage, percentage_achieved=achieved)


def generate_split_linear(start_split, end_split, ranked_operations, names, metric, percentage, weights=None):
    """Pick k as that percentage of the operation list, without searching.

    ``k = round(n_operations * percentage / 100)``. The resulting leakage is
    whatever it turns out to be, which is reported in
    ``SplitResult.percentage_achieved``.

    This is monotone in the requested percentage by construction, costs one
    leakage evaluation instead of ``log2(n_operations)``, and is the right
    choice when the sweep should step evenly through the decomposition. Use
    :func:`generate_split` instead when the split must sit at a specific
    leakage value.

    Parameters
    ----------
    start_split, end_split : dict
        Entity name -> fold at each endpoint.
    ranked_operations : list of (Operation, float)
        Output of a ranking function.
    names : list of str
        Entities, in the same order as the rows of ``metric``.
    metric : numpy.ndarray
        Square similarity matrix.
    percentage : float
        Target, on a **0-100** scale.
    weights : dict, optional
        Entity name -> weight.

    Returns
    -------
    SplitResult

    Raises
    ------
    ValueError
        When ``percentage`` is outside [0, 100].
    """
    if not (0 <= percentage <= 100):
        raise ValueError("percentage must be between 0 and 100")
    L_start = leakage(start_split, names, metric, weights)
    L_end = leakage(end_split, names, metric, weights)
    n_ops = len(ranked_operations)
    k = round(n_ops * percentage / 100)

    split = apply_k_cycles(start_split, ranked_operations, end_split, k)
    L_k = leakage(split, names, metric, weights)

    achieved = 0.0 if L_end == L_start else (L_k - L_start) / (L_end - L_start)
    return SplitResult(split=split, k=k, n_operations=n_ops,
                       leakage_start=L_start, leakage_end=L_end, leakage_achieved=L_k,
                       percentage_requested=percentage, percentage_achieved=achieved)
