"""Two-dimensional continuous splitting, by interpolating each axis separately.

In a 2-D dataset a sample is a pair, a protein and a ligand, and the fold of a
pair is constrained by both of its entities. Interpolating the pairs directly
is awkward because the pair matrix is sparse: most protein-ligand combinations
do not exist.

So each axis is interpolated on its own as an ordinary 1-D problem, and the two
results are recombined by an **agreement rule**: an interaction keeps its fold
when both of its entities landed in the same fold, and is dropped otherwise.
The fraction that survives is the *coverage*, roughly the sum of the squared
fold proportions.
"""
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .clustering import build_similarity_dataset
from .leakage import _axis_leakage
from .loaders import assert_entity_pure, load_all_rows, load_fold_labels_2d
from .split_1d import continuous_split_1D


def continuous_split_2D(
        e_start_dir: Path, e_end_dir: Path,     # protein axis, e.g. I1P -> S1P
        f_start_dir: Path, f_end_dir: Path,     # ligand  axis, e.g. I1L -> S1L
        out_dir: Path,
        e_datatype: str, e_id_col: str, e_value_col: str,
        f_datatype: str, f_id_col: str, f_value_col: str,
        percentages: List,                      # 50 -> both axes at 50 %; (25, 75) -> grid point
        e_similarity: str = "diamond",
        f_similarity: str = "ecfp",
        folds: Optional[List[str]] = None,
        e_num_clusters: Optional[int] = None,
        f_num_clusters: Optional[int] = None,
        ranking_method: str = "leakage",
        ranking_seed: Optional[int] = None,
        cache_path: Optional[Path] = None,
        k_method: str = "leakage",
        strict: bool = True,
) -> pd.DataFrame:
    """Interpolate a two-dimensional split from four one-dimensional endpoints.

    Takes two endpoints per axis, interpolates each axis independently, and
    recombines them with the agreement rule.

    Parameters
    ----------
    e_start_dir, e_end_dir : Path
        Endpoints of the first axis, conventionally the protein one. For
        example an identity-based split and a similarity-based split.
    f_start_dir, f_end_dir : Path
        Endpoints of the second axis, conventionally the ligand one.
    out_dir : Path
        Where to write the results. Created if missing.
    e_datatype, e_id_col, e_value_col : str
        DataSAIL type, identifier column and value column for the first axis.
    f_datatype, f_id_col, f_value_col : str
        The same for the second axis.
    percentages : list
        Either a number, applying the same percentage to both axes, or a pair
        ``(e_pct, f_pct)`` for an off-diagonal grid point. On a 0 to 100 scale.
    e_similarity, f_similarity : str
        Similarity measures per axis. Defaults suit proteins and molecules.
    folds : list of str, optional
        Fold names. Defaults to ``["train", "val", "test"]``.
    e_num_clusters, f_num_clusters : int, optional
        Clustering budget per axis.
    ranking_method, ranking_seed, cache_path, k_method
        Passed through to :func:`~.split_1d.continuous_split_1D`.
    strict : bool, default True
        Raise rather than warn when an endpoint is not entity-pure, or when
        the four endpoints do not cover the same interactions.

    Returns
    -------
    pandas.DataFrame
        One row per requested grid point, also written to
        ``out_dir/leakage_summary_2d_via_axes.csv``, with ``e_percentage``,
        ``f_percentage``, ``rows``, ``coverage``, ``PSL``, ``MSL``, ``TSL``,
        the per-fold-pair leakages of each axis, the reference axis leakages
        over the full row set, and the fold sizes ``n_train`` and so on.

    Notes
    -----
    Output layout::

        out_dir/
            leakage_summary_2d_via_axes.csv
            e_axis/     full 1-D sweep of the first axis
            f_axis/     full 1-D sweep of the second axis
            pct_50/train.csv ...          (both axes at the same percentage)
            e25_f75/train.csv ...         (off-diagonal grid points)

    Interactions dropped by the agreement rule are written to no fold at all,
    so the ``pct_*`` directories hold fewer rows than the input. ``coverage``
    records the fraction kept.

    Raises
    ------
    ValueError
        When ``strict`` and an endpoint is impure, or the four endpoints
        disagree about which interactions exist.
    """
    from datasail.settings import NOT_ASSIGNED

    folds = folds or ["train", "val", "test"]
    e_start_dir, e_end_dir = Path(e_start_dir), Path(e_end_dir)
    f_start_dir, f_end_dir = Path(f_start_dir), Path(f_end_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [tuple(p) if isinstance(p, (tuple, list)) else (p, p) for p in percentages]
    axis_pcts = sorted({p for pr in pairs for p in pr})

    print("Endpoint checks")
    e_prof_start = assert_entity_pure(e_start_dir, e_id_col, e_value_col, folds, "protein", strict)
    e_prof_end = assert_entity_pure(e_end_dir, e_id_col, e_value_col, folds, "protein", strict)
    f_prof_start = assert_entity_pure(f_start_dir, f_id_col, f_value_col, folds, "ligand", strict)
    f_prof_end = assert_entity_pure(f_end_dir, f_id_col, f_value_col, folds, "ligand", strict)

    for axis, a, b in [("protein", e_prof_start, e_prof_end),
                       ("ligand", f_prof_start, f_prof_end)]:
        if a == b:
            print(f"  {axis} fold sizes match {a} -- pure cycles, sizes constant across the sweep")
        else:
            print(f"  WARNING: {axis} fold sizes differ, {a} vs {b}. The decomposition will "
                  f"contain leftover singletons; entity proportions and the leakage baseline "
                  f"will drift along the sweep.")

    frames = {"e_start": load_all_rows(e_start_dir, folds, require=True),
              "e_end": load_all_rows(e_end_dir, folds, require=True),
              "f_start": load_all_rows(f_start_dir, folds, require=True),
              "f_end": load_all_rows(f_end_dir, folds, require=True)}
    keys = {k: set(map(tuple, r[[e_id_col, f_id_col]].values)) for k, r in frames.items()}
    ref = keys["f_start"]
    if any(v != ref for v in keys.values()):
        msg = ("the four splits do not cover the same interactions: "
               + ", ".join(f"{k}={len(v)}" for k, v in keys.items()))
        if strict:
            raise ValueError(msg)
        print(f"  WARNING: {msg}")
    all_rows = frames["f_start"].drop_duplicates(subset=[e_id_col, f_id_col]).reset_index(drop=True)
    print(f"  {len(all_rows)} interactions, identical across all four splits")

    # w(u) = number of interactions u takes part in -- makes Eq. (1) range over
    # interactions, which is what the paper's D is.
    e_weights = all_rows[e_id_col].value_counts().to_dict()
    f_weights = all_rows[f_id_col].value_counts().to_dict()

    # ---------------------------------------------------------------- entity labels
    e_start_points, e_start_label = load_fold_labels_2d(e_start_dir, e_id_col, e_value_col, folds)
    e_end_points, e_end_label = load_fold_labels_2d(e_end_dir, e_id_col, e_value_col, folds)
    f_start_points, f_start_label = load_fold_labels_2d(f_start_dir, f_id_col, f_value_col, folds)
    f_end_points, f_end_label = load_fold_labels_2d(f_end_dir, f_id_col, f_value_col, folds)

    e_common = set(e_start_label) & set(e_end_label)
    f_common = set(f_start_label) & set(f_end_label)
    dropped_e = (set(e_start_label) | set(e_end_label)) - e_common
    dropped_f = (set(f_start_label) | set(f_end_label)) - f_common
    if dropped_e or dropped_f:
        print(f"  WARNING: dropping {len(dropped_e)} protein(s) and {len(dropped_f)} molecule(s) "
              f"not present in both endpoints of their axis")
    e_start_points = {k: v for k, v in e_start_points.items() if k in e_common}
    e_end_points = {k: v for k, v in e_end_points.items() if k in e_common}
    e_start_label = {k: v for k, v in e_start_label.items() if k in e_common}
    e_end_label = {k: v for k, v in e_end_label.items() if k in e_common}
    f_start_points = {k: v for k, v in f_start_points.items() if k in f_common}
    f_end_points = {k: v for k, v in f_end_points.items() if k in f_common}
    f_start_label = {k: v for k, v in f_start_label.items() if k in f_common}
    f_end_label = {k: v for k, v in f_end_label.items() if k in f_common}

    # ---------------------------------------------------------------- axis projections
    # NOTE: distinct names -- these must not clobber the four input directories.
    tmp_root = Path(tempfile.mkdtemp(prefix="axis_projection_"))
    e_start_proj, e_end_proj = tmp_root / "e_start", tmp_root / "e_end"
    f_start_proj, f_end_proj = tmp_root / "f_start", tmp_root / "f_end"

    def write_axis_dir(target_dir, points, label, id_col, value_col):
        """Write a two-column 1-D split directory for one axis."""
        target_dir.mkdir(parents=True, exist_ok=True)
        rows_by_fold = {f: [] for f in folds}
        for name, fold in label.items():
            rows_by_fold[fold].append({id_col: name, value_col: points[name]})
        for fold, rows in rows_by_fold.items():
            if not rows:
                continue
            pd.DataFrame(rows).to_csv(target_dir / f"{fold}.csv", index=False)

    write_axis_dir(e_start_proj, e_start_points, e_start_label, e_id_col, e_value_col)
    write_axis_dir(e_end_proj, e_end_points, e_end_label, e_id_col, e_value_col)
    write_axis_dir(f_start_proj, f_start_points, f_start_label, f_id_col, f_value_col)
    write_axis_dir(f_end_proj, f_end_points, f_end_label, f_id_col, f_value_col)

    e_out, f_out = out_dir / "e_axis", out_dir / "f_axis"

    def _axis_cache(suffix):
        """Give each axis its own cache file, derived from cache_path."""
        if cache_path is None:
            return None
        p = Path(cache_path)
        return p.with_name(p.stem + suffix + p.suffix)

    e_cache_path, f_cache_path = _axis_cache("_e"), _axis_cache("_f")

    e_summary = continuous_split_1D(
        start_dir=e_start_proj, end_dir=e_end_proj, out_dir=e_out,
        datatype=e_datatype, id_col=e_id_col, value_col=e_value_col,
        percentages=axis_pcts, similarity=e_similarity, folds=folds,
        plots=False, curve=True, num_clusters=e_num_clusters, ranking_method=ranking_method,
        ranking_seed=ranking_seed, k_method=k_method, weights=e_weights, cache_path=e_cache_path,
    )
    f_summary = continuous_split_1D(
        start_dir=f_start_proj, end_dir=f_end_proj, out_dir=f_out,
        datatype=f_datatype, id_col=f_id_col, value_col=f_value_col,
        percentages=axis_pcts, similarity=f_similarity, folds=folds,
        plots=False, curve=True, num_clusters=f_num_clusters, ranking_method=ranking_method,
        ranking_seed=ranking_seed, k_method=k_method, weights=f_weights, cache_path=f_cache_path,
    )

    e_ds = build_similarity_dataset(e_datatype, e_start_points, similarity=e_similarity,
                                    cache_path=e_cache_path)
    f_ds = build_similarity_dataset(f_datatype, f_start_points, similarity=f_similarity,
                                    cache_path=f_cache_path)

    def load_axis_split(axis_out_dir, id_col_here, pct):
        """Read back one interpolated axis split as an entity -> fold map."""
        split = {}
        for fold in folds:
            path = axis_out_dir / f"pct_{pct}" / f"{fold}.csv"
            if not path.exists():
                continue
            for v in pd.read_csv(path)[id_col_here]:
                split[v] = fold
        return split

    e_splits = {p: load_axis_split(e_out, e_id_col, p) for p in axis_pcts}
    f_splits = {p: load_axis_split(f_out, f_id_col, p) for p in axis_pcts}

    results = []
    for e_pct, f_pct in pairs:
        e_split, f_split = e_splits[e_pct], f_splits[f_pct]
        e_col = all_rows[e_id_col].map(e_split)
        f_col = all_rows[f_id_col].map(f_split)
        # The agreement rule: keep an interaction only when both axes agree.
        agree = e_col.notna() & f_col.notna() & (e_col == f_col)
        inter_fold = np.where(agree, e_col, NOT_ASSIGNED)
        coverage = float(agree.mean())

        tag = f"pct_{e_pct}" if e_pct == f_pct else f"e{e_pct}_f{f_pct}"
        pct_dir = out_dir / tag
        pct_dir.mkdir(parents=True, exist_ok=True)
        for fold in folds:
            all_rows[inter_fold == fold].to_csv(pct_dir / f"{fold}.csv", index=False)

        # Leakage is recomputed on the surviving interactions, not on the full axis.
        kept = all_rows[agree]
        psl_pw = _axis_leakage(e_ds, e_split, set(kept[e_id_col]),
                               kept[e_id_col].value_counts().to_dict(), folds)
        msl_pw = _axis_leakage(f_ds, f_split, set(kept[f_id_col]),
                               kept[f_id_col].value_counts().to_dict(), folds)
        psl, msl = float(np.nansum(list(psl_pw.values()))), float(np.nansum(list(msl_pw.values())))

        row = {"e_percentage": e_pct, "f_percentage": f_pct, "rows": int(len(kept)),
               "coverage": coverage, "PSL": psl, "MSL": msl, "TSL": psl + msl}
        row.update({f"PSL_{k}": v for k, v in psl_pw.items()})
        row.update({f"MSL_{k}": v for k, v in msl_pw.items()})
        # leakage of the axis projections over the full row set, for reference
        row["PSL_axis"] = float(e_summary.loc[e_summary.percentage_requested == e_pct, "leakage"].iloc[0])
        row["MSL_axis"] = float(f_summary.loc[f_summary.percentage_requested == f_pct, "leakage"].iloc[0])
        for fold in folds:
            row[f"n_{fold}"] = int((inter_fold == fold).sum())
        empty = [f for f in folds if row[f"n_{f}"] == 0]
        if empty:
            print(f"  WARNING: fold(s) {empty} are empty at e={e_pct} f={f_pct}. The agreement "
                  f"rule can starve a small fold when the two axes stop aligning on it.")
        results.append(row)

        print(f"e={e_pct:>3} f={f_pct:>3} | PSL={psl:.4f} MSL={msl:.4f} TSL={psl + msl:.4f} "
              f"| rows={len(kept):>6} coverage={coverage:.3f} "
              f"| " + " ".join(f"{f}={row[f'n_{f}']}" for f in folds))

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "leakage_summary_2d_via_axes.csv", index=False)
    return results_df
