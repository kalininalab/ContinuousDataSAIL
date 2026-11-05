"""Information leakage of a split, and the ranking of interpolation steps.

Leakage is the share of total pairwise similarity in the dataset that falls
between folds rather than inside them, following Eq. (1) of Atabaigi Elmi,
Joeres and Kalinina (Brief. Bioinform. 2026):

    Leakage(s, s') = sum_{u in s, v in s'} w(u, v) / sum over all pairs w(u, v)

The denominator ranges over the whole dataset, so the pairwise values of the
three fold pairs are directly comparable with each other and add up to the
single number :func:`leakage` returns.

Weights matter whenever one entity stands for several rows. In a 2-D dataset a
protein takes part in many interactions, and the paper's denominator counts
interactions, not entities, so each entity enters with its interaction degree
as its weight.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .graph import Operation


def leakage(
        split: Dict[str, str],
        names: List[str],
        metric: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
):
    """Total leakage of a split: all cross-fold similarity over all similarity.

    Equals the sum of the three values :func:`leakage_pairwise` returns. The
    scalar counts each unordered pair in both directions over a denominator
    twice as large as the pairwise one, so the two agree exactly.

    Parameters
    ----------
    split : dict
        Entity name -> fold.
    names : list of str
        Entities, in the same order as the rows of ``metric``.
    metric : numpy.ndarray
        Square similarity matrix.
    weights : dict, optional
        Entity name -> weight. Defaults to 1 for every entity.

    Returns
    -------
    float
        Leakage in [0, 1], or 0.0 when the denominator vanishes.
    """
    folds = np.array([split[name] for name in names])
    w = np.ones(len(names)) if weights is None else np.array([weights.get(n, 1.0) for n in names])
    # mask[i, j] == 1 exactly when i and j share a fold.
    mask = np.zeros((len(names), len(names)))
    for f in set(folds.tolist()):
        v = (folds == f).astype(float).reshape(-1, 1)
        mask += v @ v.T
    # Subtracting w.sum() drops the diagonal, where every entity matches itself.
    den = float(w @ metric @ w - w.sum())
    num = float(np.sum((1 - mask) * (w.reshape(-1, 1) @ w.reshape(1, -1)) * metric))
    return 0.0 if den <= 0 else num / den


def leakage_pairwise(
        split: Dict[str, str],
        names: List[str],
        metric: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
        fold_pairs=(("train", "test"), ("train", "val"), ("val", "test")),
) -> Dict[str, float]:
    """Per-pair-of-folds leakage, directly comparable to Table 3 of the paper.

    Parameters
    ----------
    split : dict
        Entity name -> fold.
    names : list of str
        Entities, in the same order as the rows of ``metric``.
    metric : numpy.ndarray
        Square similarity matrix.
    weights : dict, optional
        Entity name -> weight. Defaults to 1 for every entity.
    fold_pairs : tuple of (str, str)
        Which pairs to report. A pair naming a fold that does not exist in
        ``split`` yields 0.0 rather than an error, so two-fold splits work.

    Returns
    -------
    dict
        ``"train-test"`` and friends -> leakage. The denominator is shared
        across the pairs, so the values sum to :func:`leakage`.
    """
    folds = np.array([split[name] for name in names])
    w = np.ones(len(names)) if weights is None else np.array([weights.get(n, 1.0) for n in names])
    # Half of leakage()'s denominator, because each unordered pair is counted once here.
    den = (float(w @ metric @ w) - float(w.sum())) / 2.0
    out = {}
    for s, sp in fold_pairs:
        a, b = (folds == s) * w, (folds == sp) * w
        out[f"{s}-{sp}"] = 0.0 if den <= 0 else float(a @ metric @ b) / den
    return out


def _fold_pairs(folds):
    """Every unordered pair of folds, in the order ``folds`` gives them."""
    return tuple((a, b) for i, a in enumerate(folds) for b in folds[i + 1:])


def _axis_leakage(dataset, split, kept_ids, weights, folds):
    """Eq. (1) on the interactions that were actually written out, at entity resolution.

    The agreement rule of the 2-D split discards interactions whose two axes
    disagree, so leakage has to be recomputed on the entities that survive
    rather than on the full axis projection.

    Parameters
    ----------
    dataset : datasail DataSet
        Carries ``cluster_names`` and ``cluster_similarity`` for this axis.
    split : dict
        Entity name -> fold for this axis.
    kept_ids : set
        Entities still present after the agreement rule.
    weights : dict
        Entity name -> number of surviving interactions it takes part in.
    folds : list of str
        Fold names.

    Returns
    -------
    dict
        Pair name -> leakage, all NaN when fewer than two entities survive.
    """
    order = {n: i for i, n in enumerate(dataset.cluster_names)}
    names = [n for n in dataset.cluster_names if n in split and n in kept_ids]
    if len(names) < 2:
        return {f"{a}-{b}": float("nan") for a, b in _fold_pairs(folds)}
    pos = [order[n] for n in names]
    metric = dataset.cluster_similarity[np.ix_(pos, pos)]
    return leakage_pairwise({n: split[n] for n in names}, names, metric,
                            weights, fold_pairs=_fold_pairs(folds))


def rank_operations(
        strata_cycles: Dict[Tuple, Tuple[List[Operation], List[str]]],
        start_split: Dict[str, str],
        end_split: Dict[str, str],
        names: List[str],
        metric: "np.ndarray",
        weights: Dict[str, float] = None,):
    """Order the interpolation steps by how much leakage each one adds.

    Every operation is scored on its own against the start split: delta_i is
    the change in cross-fold similarity caused by applying operation i and
    nothing else. Sorting ascending puts the operations that lower leakage, or
    raise it least, first. Applying the first k of them therefore walks the
    split away from its starting leakage as slowly as the decomposition allows.

    The deltas are not recomputed as operations accumulate, so they are exact
    only for the first step; after that they are an ordering heuristic. This is
    what gives the leakage curve its characteristic early dip.

    Parameters
    ----------
    strata_cycles : dict
        Key -> ``(cycles, leftover)``, as returned by :func:`~.graph.decompose`.
        Callers that do not separate strata pass a single arbitrary key.
    start_split, end_split : dict
        Entity name -> fold at each endpoint.
    names : list of str
        Entities, in the same order as the rows of ``metric``.
    metric : numpy.ndarray
        Square similarity matrix.
    weights : dict, optional
        Entity name -> weight. Defaults to 1 for every entity.

    Returns
    -------
    list of (Operation, float)
        Operations paired with their delta, sorted by delta ascending.
    """
    id_to_pos = {name: i for i, name in enumerate(names)}
    folds = np.array([start_split[name] for name in names])
    end_folds = np.array([end_split[name] for name in names])
    w = np.ones(len(names)) if weights is None else np.array([weights.get(n, 1.0) for n in names])

    def delta_for_positions(pos, new_folds_at_pos):
        """Change in cross-fold similarity from moving the entities at ``pos``."""
        row = metric[pos] * w[None, :]
        own_w = w[pos][:, None]
        new_term = ((new_folds_at_pos[:, None] != folds) * row * own_w).sum()
        old_term = ((folds[pos][:, None] != folds) * row * own_w).sum()
        return new_term - old_term

    results: List[Tuple[Operation, float]] = []
    for cycles, leftover in strata_cycles.values():
        for cycle in cycles:
            pos = np.array([id_to_pos[n] for n in cycle])
            # Rotating a cycle shifts each member into the next member's fold.
            new_folds_at_pos = np.roll(folds[pos], -1)
            results.append((cycle, delta_for_positions(pos, new_folds_at_pos)))
        for name in leftover:
            pos = np.array([id_to_pos[name]])
            new_folds_at_pos = np.array([end_folds[pos[0]]])
            results.append(((name,), delta_for_positions(pos, new_folds_at_pos)))

    return sorted(results, key=lambda cr: cr[1])


def rank_operations_random(
        strata_cycles: Dict[Tuple, Tuple[List[Operation], List[str]]],
        start_split: Dict[str, str] = None,
        end_split: Dict[str, str] = None,
        names: List[str] = None,
        metric: "np.ndarray" = None,
        weights: Dict[str, float] = None,
        seed: Optional[int] = None,
):
    """Order the interpolation steps at random.

    The baseline against which leakage ranking is judged. Takes the same
    arguments as :func:`rank_operations` so the two are interchangeable, but
    ignores everything except ``strata_cycles`` and ``seed``.

    Parameters
    ----------
    strata_cycles : dict
        Key -> ``(cycles, leftover)``.
    start_split, end_split, names, metric, weights
        Accepted and ignored, for signature compatibility.
    seed : int, optional
        Seed for the shuffle. Pass one to make a run reproducible.

    Returns
    -------
    list of (Operation, float)
        Operations in random order, each paired with a placeholder delta of 0.0.
    """
    # Random ranking discards the leakage-delta immediately (shuffle, not sort),
    # so skip computing it -- this used to redo the same O(n^2)-ish work as
    # rank_operations for a value nobody reads.
    results: List[Tuple[Operation, float]] = []
    for cycles, leftover in strata_cycles.values():
        for cycle in cycles:
            results.append((cycle, 0.0))
        for name in leftover:
            results.append(((name,), 0.0))

    rng = np.random.default_rng(seed)
    rng.shuffle(results)
    return results
