"""Continuous interpolation between two DataSAIL splits.

DataSAIL produces splits with minimal information leakage, but models trained
on them score worse than models trained on random splits. Rather than choosing
between the two extremes, this package builds the splits that lie between them,
so the trade-off can be measured instead of guessed.

How it works
------------
A split is a map from entity to fold. Two splits of the same entities define a
directed multigraph whose nodes are folds and whose edges are the entities that
change fold. Decomposing that graph into cycles gives a list of operations,
each of which can be applied on its own without changing any fold's size.
Ordering those operations by their effect on leakage and applying the first k
produces a split a controllable distance from each endpoint.

Entry points
------------
:func:`continuous_split_1D`
    Interpolate between two splits of single entities.
:func:`continuous_split_2D`
    Interpolate a split of interaction pairs, by interpolating each axis and
    recombining them with an agreement rule.

Layout
------
``loaders``
    Reading split directories.
``graph``
    The transition graph and its cycle decomposition.
``leakage``
    Eq. (1), its per-fold-pair form, and the ranking of operations.
``apply``
    Applying operations, and choosing how many to apply.
``clustering``
    Similarity matrices, and grouping entities for large datasets.
``plots``
    The leakage curve.
``split_1d``, ``split_2d``
    The two entry points.

Note
----
``percentages`` is on a 0 to 100 scale everywhere, not 0 to 1.
"""
from .apply import SplitResult, apply_cycle, apply_k_cycles, generate_split, generate_split_linear
from .clustering import build_similarity_dataset, cluster_precomputed_matrix, cluster_within_groups
from .graph import Operation, build_split_graph, decompose
from .leakage import leakage, leakage_pairwise, rank_operations, rank_operations_random
from .loaders import assert_entity_pure, load_all_rows, load_fold_labels, load_fold_labels_2d
from .plots import plot_leakage_curve
from .split_1d import continuous_split_1D
from .split_2d import continuous_split_2D

__version__ = "0.2.0"

__all__ = [
    "continuous_split_1D",
    "continuous_split_2D",
    "leakage",
    "leakage_pairwise",
    "rank_operations",
    "rank_operations_random",
    "build_split_graph",
    "decompose",
    "Operation",
    "apply_cycle",
    "apply_k_cycles",
    "generate_split",
    "generate_split_linear",
    "SplitResult",
    "build_similarity_dataset",
    "cluster_precomputed_matrix",
    "cluster_within_groups",
    "assert_entity_pure",
    "load_all_rows",
    "load_fold_labels",
    "load_fold_labels_2d",
    "plot_leakage_curve",
]
