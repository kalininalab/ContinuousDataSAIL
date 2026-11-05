"""Reading DataSAIL split directories.

A *split directory* holds one delimited file per fold, named after the fold:
``train.csv``, ``val.csv``, ``test.csv`` (``.tsv`` also works). Every function
here turns such a directory into plain dictionaries and pandas frames. Nothing
in this module knows about leakage or interpolation.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _find_fold_file(split_dir, fold):
    """Locate the file holding one fold, trying ``.csv`` first and then ``.tsv``.

    Parameters
    ----------
    split_dir : str or Path
        Directory holding one file per fold.
    fold : str
        Fold name, for example ``"train"``.

    Returns
    -------
    (Path, str) or (None, None)
        The file and the separator it uses, or ``(None, None)`` when the fold
        has no file. A missing fold is not an error: two-fold splits with only
        train and test are common.
    """
    for ext, sep in [(".csv", ","), (".tsv", "\t")]:
        path = Path(split_dir) / f"{fold}{ext}"
        if path.exists():
            return path, sep
    return None, None


def load_fold_labels(split_dir, id_col, value_col, folds):
    """Map every entity in a split directory to its fold and to its value.

    This is the one-dimensional reader: it assumes each entity appears in
    exactly one fold, which is what a 1-D split guarantees. Duplicate IDs
    inside a single fold file are collapsed before reading.

    Parameters
    ----------
    split_dir : str or Path
        Directory holding one file per fold.
    id_col : str
        Column holding the entity identifier.
    value_col : str
        Column holding the entity itself, i.e. a SMILES string for molecules or
        an amino-acid sequence for proteins.
    folds : list of str
        Fold names to look for.

    Returns
    -------
    (dict, dict)
        ``points`` mapping id -> value, and ``label`` mapping id -> fold.
    """
    label, pts = {}, {}
    for fold in folds:
        path, sep = _find_fold_file(split_dir, fold)
        if path is None:
            continue
        # One row per entity: a 1-D split cannot put the same entity twice.
        df = pd.read_csv(path, sep=sep).drop_duplicates(subset=id_col)
        for _, row in df.iterrows():
            label[row[id_col]] = fold
            pts[row[id_col]] = row[value_col]
    return pts, label


def load_all_rows(split_dir, folds, require=False):
    """Concatenate every fold file of a split directory back into one frame.

    The interpolated splits are written by re-partitioning these rows, so all
    original columns (labels, measurements, anything else) survive the round
    trip untouched.

    Parameters
    ----------
    split_dir : str or Path
        Directory holding one file per fold.
    folds : list of str
        Fold names to look for.
    require : bool, default False
        Raise when the directory holds no fold file at all, instead of
        returning an empty frame.

    Returns
    -------
    pandas.DataFrame
        Every row of every fold, in fold order, with a fresh index.
    """
    parts = []
    for f in folds:
        path, sep = _find_fold_file(split_dir, f)
        if path is not None:
            parts.append(pd.read_csv(path, sep=sep))
    if not parts:
        if require:
            raise ValueError(f"no fold files found in {split_dir}")
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def load_fold_labels_2d(
        split_dir: Path,
        id_col: str,
        value_col: str,
        folds: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Project one axis of a two-dimensional split down to entity labels.

    In a 2-D split a row is an interaction, so one protein can appear in many
    rows and in principle in several folds. This reader assigns each entity the
    fold it appears in most often. For an axis that is genuinely entity-pure
    the vote is unanimous and the majority rule changes nothing; use
    :func:`assert_entity_pure` to check that before relying on it.

    Parameters
    ----------
    split_dir : Path
        Directory holding one file per fold.
    id_col : str
        Column holding the identifier of the axis being projected.
    value_col : str
        Column holding the entity itself for that axis.
    folds : list of str, optional
        Fold names to look for. Defaults to ``["train", "val", "test"]``.

    Returns
    -------
    (dict, dict)
        ``points`` mapping id -> value, and ``label`` mapping id -> fold.
    """
    from collections import Counter
    folds = folds or ["train", "val", "test"]
    split_dir = Path(split_dir)
    votes: Dict[str, "Counter"] = {}
    pts: Dict[str, str] = {}
    for fold in folds:
        path, sep = _find_fold_file(split_dir, fold)
        if path is None:
            continue
        df = pd.read_csv(path, sep=sep)
        for _, row in df.iterrows():
            eid = row[id_col]
            votes.setdefault(eid, Counter())[fold] += 1
            pts[eid] = row[value_col]
    label = {eid: counter.most_common(1)[0][0] for eid, counter in votes.items()}
    return pts, label


def assert_entity_pure(split_dir, id_col, value_col, folds, axis_name, strict=True):
    """Check that no entity of one axis appears in more than one fold.

    Interpolating towards a split only makes sense when that split assigns each
    entity a single fold. If it does not, the entity-to-fold map would have to
    be invented by majority vote, and the endpoint of the interpolation would
    not be the split the user asked for.

    Purity is tested on ``value_col``, not ``id_col``: two rows carrying the
    same SMILES under different IDs are the same molecule, and splitting them
    apart is leakage even though the identifiers differ.

    Parameters
    ----------
    split_dir : str or Path
        Directory holding one file per fold.
    id_col : str
        Unused, kept so callers can pass the same arguments as the loaders.
    value_col : str
        Column holding the entity itself. Purity is judged on this column.
    folds : list of str
        Fold names to look for.
    axis_name : str
        Name used in messages, for example ``"protein"`` or ``"ligand"``.
    strict : bool, default True
        Raise on impurity instead of printing a warning.

    Returns
    -------
    dict
        Fold name -> number of entities that live in that fold alone. Two
        endpoints with identical profiles produce a balanced transition graph,
        which keeps fold sizes constant along the sweep.

    Raises
    ------
    ValueError
        When ``strict`` and at least one entity spans several folds.
    """
    split_dir = Path(split_dir)
    seen: Dict[str, set] = {}
    for fold in folds:
        path, sep = _find_fold_file(split_dir, fold)
        if path is None:
            continue
        for v in pd.read_csv(path, sep=sep)[value_col]:
            seen.setdefault(v, set()).add(fold)
    bad = [v for v, fs in seen.items() if len(fs) > 1]
    profile = {f: sum(1 for fs in seen.values() if len(fs) == 1 and f in fs) for f in folds}
    if bad:
        msg = (f"{split_dir.name} is not a {axis_name} split: {len(bad)} of {len(seen)} "
               f"entities appear in more than one fold. Interpolating towards it would "
               f"require inventing an entity->fold map by majority vote.")
        if strict:
            raise ValueError(msg)
        print(f"  WARNING: {msg}")
    print(f"  {split_dir.name:>8} {axis_name:>8}: {len(seen)} entities, "
          + " ".join(f"{f} {profile[f]}" for f in folds)
          + (f", {len(bad)} impure" if bad else ""))
    return profile
