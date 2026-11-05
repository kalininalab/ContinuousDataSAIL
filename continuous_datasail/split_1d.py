"""One-dimensional continuous splitting: the main entry point.

A 1-D split assigns folds to single entities, molecules or proteins, rather
than to pairs. Given two such splits of the same entities, this module produces
a family of splits lying between them.
"""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .apply import generate_split, generate_split_linear
from .clustering import build_similarity_dataset, cluster_within_groups
from .graph import build_split_graph, decompose
from .leakage import leakage_pairwise, rank_operations, rank_operations_random
from .loaders import load_all_rows, load_fold_labels
from .plots import plot_leakage_curve


def continuous_split_1D(
    start_dir: Path,
    end_dir: Path,
    out_dir: Path,
    datatype: str,
    id_col: str,
    value_col: str,
    percentages: List[float],
    similarity: str = "ecfp",
    folds: Optional[List[str]] = None,
    plots: bool = True,
    curve: bool = True,
    weights: Optional[Dict[str, float]] = None,
    num_clusters: Optional[int] = None,
    ranking_method: str = "leakage",
    ranking_seed: Optional[int] = None,
    cache_path: Optional[Path] = None,
    k_method: str = "leakage",
    dataset=None,
) -> pd.DataFrame:
    """Interpolate between two one-dimensional splits of the same entities.

    Reads both endpoints, builds the fold-transition graph, decomposes it into
    cycles, ranks those cycles, and writes out one split per requested
    percentage.

    Parameters
    ----------
    start_dir : Path
        Split to interpolate **from**, one file per fold. Usually the
        leakage-minimised DataSAIL split.
    end_dir : Path
        Split to interpolate **to**, over exactly the same entities. Usually a
        random or identity-based split.
    out_dir : Path
        Where to write the results. Created if missing.
    datatype : str
        DataSAIL data type: ``"M"`` for molecules, ``"P"`` for proteins.
    id_col : str
        Column holding the entity identifier.
    value_col : str
        Column holding the entity itself: a SMILES string or a sequence.
    percentages : list of float
        Requested positions along the interpolation, **on a 0 to 100 scale**.
        ``[0, 10, ..., 100]``, not ``[0, 0.1, ..., 1]``. Passing fractions is
        accepted but means "the first one percent of the interpolation", which
        produces a set of near-identical splits.
    similarity : str, default "ecfp"
        Similarity measure: ``"ecfp"`` for molecules, ``"diamond"`` for proteins.
    folds : list of str, optional
        Fold names to look for. Defaults to ``["train", "val", "test"]``. Pass
        ``["train", "test"]`` for a two-fold split.
    plots : bool, default True
        Write a t-SNE plot of each interpolated split, via DataSAIL.
    curve : bool, default True
        Write ``leakage_vs_k.png``, the leakage curve of the whole sweep.
    weights : dict, optional
        Entity name -> weight. Defaults to the number of rows each entity
        occupies, which makes Eq. (1) range over rows rather than entities.
    num_clusters : int, optional
        Group entities into roughly this many clusters before interpolating,
        to make large datasets tractable. ``None`` interpolates over entities.
    ranking_method : {"leakage", "random"}, default "leakage"
        How to order the operations.
    ranking_seed : int, optional
        Seed for ``ranking_method="random"``.
    cache_path : Path, optional
        Pickle the similarity matrix here and reuse it on later runs. The file
        is about ``n**2 * 8`` bytes; prefer ``dataset`` when several runs share
        one process.
    k_method : {"leakage", "linear"}, default "leakage"
        How to turn a percentage into a number of operations. ``"leakage"``
        binary-searches for the target leakage; ``"linear"`` takes that
        fraction of the operation list. See :func:`~.apply.generate_split`.
    dataset : datasail DataSet, optional
        Pre-built similarity dataset, skipping ``build_similarity_dataset``
        entirely.

    Returns
    -------
    pandas.DataFrame
        One row per requested percentage, also written to
        ``out_dir/leakage_summary.csv``, with columns:

        ``percentage_requested``
            What was asked for, 0-100.
        ``percentage_achieved``
            Where the split landed, as a fraction 0-1 of the way between the
            endpoint leakages.
        ``leakage``
            Total leakage of this split.
        ``k``, ``n_operations``
            Operations applied, out of the whole decomposition.
        ``leakage_train-test``, ``leakage_train-val``, ``leakage_val-test``
            Per-fold-pair leakage. These sum to ``leakage``.

    Notes
    -----
    Output layout::

        out_dir/
            leakage_summary.csv
            leakage_vs_k.png        (curve=True)
            pct_0/train.csv  val.csv  test.csv
            pct_10/...
            ...

    Each ``pct_*`` directory holds the original rows, re-partitioned. Every
    column of the input survives.

    Fold sizes stay constant along the sweep only when the two endpoints give
    each fold the same number of entities. Otherwise the decomposition contains
    leftovers, which change fold sizes as they are applied.
    """
    from datasail.report import save_t_sne

    folds = folds or ["train", "val", "test"]
    start_dir, end_dir, out_dir = Path(start_dir), Path(end_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(percentages) > 1 and all(p <= 1 for p in percentages):
        print("  WARNING: every requested percentage is <= 1. percentages is on a 0-100 "
              "scale, so this explores only the first 1% of the interpolation and the "
              "resulting splits will be nearly identical. Did you mean "
              f"{[p * 100 for p in percentages]}?")

    start_points, label_start = load_fold_labels(start_dir, id_col, value_col, folds)
    end_points, label_end = load_fold_labels(end_dir, id_col, value_col, folds)
    assert set(start_points) == set(end_points), "start_dir and end_dir must cover the same entities"
    points = start_points

    # Union of both endpoints' rows: the interpolated splits are written by
    # re-partitioning these, so no input row can be invented or lost.
    all_rows = pd.concat([load_all_rows(start_dir, folds), load_all_rows(end_dir, folds)],
                         ignore_index=True).drop_duplicates()
    if weights is None:
        weights = all_rows[id_col].value_counts().to_dict()
    if dataset is None:
        dataset = build_similarity_dataset(datatype, points, similarity=similarity, cache_path=cache_path)
    names, metric = dataset.cluster_names, dataset.cluster_similarity

    if num_clusters is not None and num_clusters < len(names):
        (interp_names, cluster_map, interp_metric, interp_weights,
         interp_label_start, interp_label_end) = cluster_within_groups(
             names, metric, label_start, label_end, total_clusters=num_clusters,
             entity_weights=weights,
             )
    else:
        cluster_map = {n: n for n in names}
        interp_names, interp_metric, interp_weights = names, metric, weights
        interp_label_start, interp_label_end = label_start, label_end

    # --- Restore-dropped-rows guard: whatever entities disappear during dataset
    # construction (e.g. duplicate-molecule collapsing), map them back onto a
    # surviving structural duplicate so no input row is silently lost downstream.
    dropped_ids = set(points) - set(names)
    restore_map = {}
    if dropped_ids:
        if datatype == "M":
            from rdkit import Chem

            def canon_key(smiles):
                mol = Chem.MolFromSmiles(smiles)
                return Chem.MolToInchi(mol) if mol is not None else smiles
        else:
            def canon_key(value):
                return value  # exact-value grouping for non-molecule datatypes

        key_to_survivor = {}
        for n in names:
            key_to_survivor.setdefault(canon_key(points[n]), n)
        for did in dropped_ids:
            key = canon_key(points[did])
            if key in key_to_survivor:
                restore_map[did] = key_to_survivor[key]
        print(f"Restoring {len(restore_map)}/{len(dropped_ids)} dropped entities via structural duplicates "
              f"({len(dropped_ids) - len(restore_map)} had no surviving duplicate; falling back to start label)")

    graph = build_split_graph(interp_label_start, interp_label_end, interp_names)
    cycles, leftover = decompose(graph)
    if leftover:
        print(f"  {len(cycles)} cycle(s), {len(leftover)} leftover(s): fold sizes change along the sweep")
    else:
        print(f"  {len(cycles)} cycle(s), no leftovers: fold sizes stay constant along the sweep")

    # rank_operations takes a dict so it can one day rank several strata
    # together; with no stratification there is a single arbitrary key.
    if ranking_method == "leakage":
        ranked = rank_operations({(1, 1): (cycles, leftover)}, interp_label_start, interp_label_end,
                                 interp_names, interp_metric, interp_weights)
    elif ranking_method == "random":
        ranked = rank_operations_random({(1, 1): (cycles, leftover)}, interp_label_start, interp_label_end,
                                        interp_names, interp_metric, interp_weights, seed=ranking_seed)
    else:
        raise ValueError(f"Unknown ranking_method: {ranking_method!r}. Use 'leakage' or 'random'.")

    if curve:
        plot_leakage_curve(interp_label_start, interp_label_end, ranked, interp_names, interp_metric,
                           out_dir / "leakage_vs_k.png", weights=interp_weights)

    fold_names = sorted(set(interp_label_start.values()) | set(interp_label_end.values()))
    results = []
    for pct in percentages:
        if k_method == "leakage":
            result = generate_split(interp_label_start, interp_label_end, ranked, interp_names,
                                    interp_metric, percentage=pct, weights=interp_weights)
        elif k_method == "linear":
            result = generate_split_linear(interp_label_start, interp_label_end, ranked, interp_names,
                                           interp_metric, percentage=pct, weights=interp_weights)
        else:
            raise ValueError(f"Unknown k_method: {k_method!r}. Use 'leakage' or 'linear'.")

        # Clusters carry their members with them; with no clustering this is the identity.
        entity_split = {n: result.split[cluster_map[n]] for n in names}

        # backfill entities that were dropped from the dataset
        for did in dropped_ids:
            if did in restore_map:
                entity_split[did] = entity_split[restore_map[did]]
            else:
                entity_split[did] = label_start[did]  # last-resort fallback: keep original fold

        pct_dir = out_dir / f"pct_{pct}"
        pct_dir.mkdir(parents=True, exist_ok=True)
        row_folds = all_rows[id_col].map(entity_split)
        for fold in fold_names:
            all_rows[row_folds == fold].to_csv(pct_dir / f"{fold}.csv", index=False)

        if plots:
            save_t_sne(pct_dir, dataset, entity_split, entity_split, fold_names)

        pw = leakage_pairwise(result.split, interp_names, interp_metric, interp_weights)
        results.append({"percentage_requested": pct, "percentage_achieved": result.percentage_achieved,
                        "leakage": result.leakage_achieved, "k": result.k,
                        "n_operations": result.n_operations,
                        **{f"leakage_{k}": v for k, v in pw.items()}})
        print(f"pct={pct:>3} | k={result.k:>4}/{result.n_operations} | leakage={result.leakage_achieved:.4f} "
              f"| train-test={pw['train-test']:.4f} train-val={pw['train-val']:.4f} val-test={pw['val-test']:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "leakage_summary.csv", index=False)
    return results_df
