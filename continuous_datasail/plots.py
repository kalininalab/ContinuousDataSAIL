"""Plotting the leakage curve.

Matplotlib is forced onto the ``Agg`` backend at import time, because these
functions run on cluster nodes with no display. Figures are created and closed
through the object-oriented interface rather than the pyplot global state:
DataSAIL's own ``save_matrix_tsne`` draws on whatever figure pyplot has active,
so a figure left open here would end up underneath the next t-SNE plot.
"""
import matplotlib
matplotlib.use("Agg")  # headless cluster nodes have no $DISPLAY; force a non-interactive backend
import matplotlib.pyplot as plt

from .apply import apply_cycle
from .leakage import leakage


def plot_leakage_curve(start_split, end_split, ranked_operations, names, metric, out_path, weights=None, max_points=300):
    """Plot leakage against the number of applied operations.

    Walks the whole ranked list, applying every operation in order, and records
    leakage at evenly spaced points along the way. The two endpoint leakages
    are drawn as horizontal reference lines.

    The shape of this curve is the main diagnostic for a run. Under leakage
    ranking it typically dips before rising, because the cheapest operations
    reduce leakage below the starting split; under random ranking it is closer
    to a straight line.

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
    out_path : str or Path
        Where to write the PNG.
    weights : dict, optional
        Entity name -> weight.
    max_points : int, default 300
        How many points to evaluate along the curve. Only the resolution of the
        picture depends on this; every operation is applied regardless, so the
        split the walk ends on is unaffected.
    """
    # leakage() rebuilds a full n x n matrix every call. With thousands of
    # operations that dominates runtime for BOTH ranking methods. Sample at
    # most max_points of them -- apply_cycle still runs for every operation,
    # so the final split is unaffected, only curve resolution changes.
    n_ops = len(ranked_operations)
    stride = max(1, n_ops // max_points)

    split = dict(start_split)
    ks = [0]
    leakages = [leakage(split, names, metric, weights)]
    for i, (operation, _) in enumerate(ranked_operations, start=1):
        split = apply_cycle(split, operation, end_split)
        if i % stride == 0 or i == n_ops:
            ks.append(i)
            leakages.append(leakage(split, names, metric, weights))
    L_start = leakage(start_split, names, metric, weights)
    L_end = leakage(end_split, names, metric, weights)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, leakages, "-", lw=1.5)
    ax.axhline(L_start, color="#e34948", lw=1.5, ls="--", label="L(start_split)")
    ax.axhline(L_end, color="#1baf7a", lw=1.5, ls="--", label="L(end_split)")
    ax.set_xlabel("k (number of ranked operations applied)")
    ax.set_ylabel("Leakage")
    ax.set_title("Leakage vs. number of applied cycle operations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)   # never leave a figure open: the next t-SNE would draw onto it
