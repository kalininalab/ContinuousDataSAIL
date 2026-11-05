"""Similarity matrices, and grouping entities to make large datasets tractable.

Every leakage evaluation touches the full n-by-n similarity matrix, so cost
grows quadratically with the number of entities. Clustering replaces entities
with groups and runs the interpolation on those instead, which shrinks n.

The grouping is constrained: only entities that share both a start fold and an
end fold may land in the same cluster. A cluster is then unambiguous at both
endpoints, carries the summed weight of its members, and the rest of the
algorithm applies to clusters exactly as it would to entities.

The similarity matrix is always computed on the raw entities first and only
then aggregated, which costs memory but avoids measuring similarity between
things that are already approximations.
"""
import os
from typing import Dict, List, Tuple

import numpy as np


def build_similarity_dataset(datatype: str, data, similarity="ecfp", cache_path=None):
    """Compute the all-against-all similarity matrix through DataSAIL.

    Parameters
    ----------
    datatype : str
        DataSAIL data type, ``"M"`` for molecules or ``"P"`` for proteins.
    data : dict
        Entity name -> value (SMILES string, amino-acid sequence, ...).
    similarity : str, default "ecfp"
        Similarity measure. ``"ecfp"`` for molecules, ``"diamond"`` or another
        aligner for proteins.
    cache_path : str or Path, optional
        Where to pickle the result. An existing file is loaded instead of
        recomputing. The pickle holds the full matrix, so it is roughly
        ``n**2 * 8`` bytes: a few hundred megabytes for ten thousand entities.
        Pass ``dataset=`` to the split functions instead when several runs share
        one matrix inside a single process.

    Returns
    -------
    datasail DataSet
        With ``cluster_names`` and ``cluster_similarity`` populated.
    """
    import pickle
    from pathlib import Path as _Path

    if cache_path is not None and _Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from datasail.reader.read import read_data_type
    from datasail.cluster.clustering import cluster

    dataset = read_data_type(datatype)(
        data=data, weights=None, sim=similarity, dist=None,
        num_clusters=np.inf, detect_duplicates=False,
    )
    dataset.id_map = {n: n for n in dataset.names}   # must stay before cluster() -- see earlier tox21 fix
    dataset = cluster(dataset, threads=os.cpu_count() or 1, logdir=None, linkage="average", output=None)

    if cache_path is not None:
        cache_path = _Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(dataset, f)
        tmp_path.replace(cache_path)  # atomic rename, safe even if another job is mid-write elsewhere

    return dataset


def cluster_precomputed_matrix(
        names: List[str],
        metric: np.ndarray,
        num_clusters: int,
        linkage: str = "average",
        prefix: str = "",
) -> Tuple[List[str], Dict[str, str], np.ndarray]:
    """Agglomerative clustering of an existing similarity matrix, via DataSAIL.

    Parameters
    ----------
    names : list of str
        Entities, in the same order as the rows of ``metric``.
    metric : numpy.ndarray
        Square similarity matrix.
    num_clusters : int
        How many clusters to reduce to.
    linkage : str, default "average"
        Linkage rule passed through to DataSAIL.
    prefix : str, default ""
        Prepended to every cluster name, so clusters built from different
        groups of entities cannot collide.

    Returns
    -------
    (list of str, dict, numpy.ndarray)
        Cluster names, entity name -> cluster name, and the cluster-level
        similarity matrix.
    """
    from datasail.reader.utils import DataSet
    from datasail.cluster.clustering import additional_clustering

    dataset = DataSet(
        names=list(names), cluster_names=list(names),
        cluster_map={n: n for n in names},
        cluster_weights={n: 1.0 for n in names},
        num_clusters=num_clusters,
        cluster_similarity=metric,
    )
    reduced = additional_clustering(dataset, n_clusters=num_clusters, linkage=linkage)
    cluster_names = [f"{prefix}{c}" for c in reduced.cluster_names]
    cluster_map = {orig: f"{prefix}{c}" for orig, c in reduced.cluster_map.items()}
    return cluster_names, cluster_map, reduced.cluster_similarity


def cluster_within_groups(names, metric, label_start, label_end, total_clusters=50,
                          linkage="average", entity_weights=None):
    """Cluster entities without mixing different start/end fold transitions.

    Entities are first partitioned by the pair ``(start fold, end fold)``, and
    clustering happens only inside each part. A cluster therefore has one start
    fold and one end fold, which is what lets the transition graph be built on
    clusters rather than entities.

    The cluster budget is split proportionally to part size. A part smaller
    than its budget is left alone, each of its entities becoming its own
    cluster, and rounding drift is absorbed by the largest part so the totals
    come out right.

    Parameters
    ----------
    names : list of str
        Entities, in the same order as the rows of ``metric``.
    metric : numpy.ndarray
        Square similarity matrix over ``names``.
    label_start, label_end : dict
        Entity name -> fold at each endpoint.
    total_clusters : int, default 50
        Target number of clusters across all parts.
    linkage : str, default "average"
        Linkage rule, also used to aggregate the cluster-level similarity:
        average of members for ``"average"``, max for ``"single"``, min for
        ``"complete"``.
    entity_weights : dict, optional
        Entity name -> weight. A cluster's weight is the sum of its members',
        which keeps Eq. (1) ranging over the same population it did before
        clustering.

    Returns
    -------
    tuple
        ``(cluster_names, cluster_map, cluster_metric, cluster_weights,
        cluster_label_start, cluster_label_end)``, where ``cluster_map`` takes
        an entity name to its cluster name.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    groups: Dict[Tuple[str, str], List[str]] = {}
    for n in names:
        groups.setdefault((label_start[n], label_end[n]), []).append(n)

    raw = {k: total_clusters * len(v) / len(names) for k, v in groups.items()}
    budget = {k: max(1, round(b)) for k, b in raw.items()}
    drift = total_clusters - sum(budget.values())
    if drift != 0:
        biggest = max(groups, key=lambda k: len(groups[k]))
        budget[biggest] = max(1, budget[biggest] + drift)

    full_cluster_map: Dict[str, str] = {}
    for key, members in groups.items():
        if len(members) <= budget[key]:
            for m in members:
                full_cluster_map[m] = m  # too small to reduce -- each entity is its own cluster
        else:
            sub_idx = [name_to_idx[m] for m in members]
            sub_metric = metric[np.ix_(sub_idx, sub_idx)]
            _, group_map, _ = cluster_precomputed_matrix(
                members, sub_metric, budget[key], linkage=linkage, prefix=f"{key[0]}-{key[1]}::",
            )
            full_cluster_map.update(group_map)

    cluster_names = sorted(set(full_cluster_map.values()))
    members_of = {c: [] for c in cluster_names}
    for n in names:
        members_of[full_cluster_map[n]].append(name_to_idx[n])

    agg_fn = {"average": np.mean, "single": np.max, "complete": np.min}[linkage]
    k = len(cluster_names)
    cluster_metric = np.zeros((k, k))
    for a, ca in enumerate(cluster_names):
        for b, cb in enumerate(cluster_names):
            cluster_metric[a, b] = agg_fn(metric[np.ix_(members_of[ca], members_of[cb])])

    ew = (np.ones(len(names)) if entity_weights is None
          else np.array([entity_weights.get(n, 1.0) for n in names]))
    cluster_weights = {c: float(ew[members_of[c]].sum()) for c in cluster_names}
    cluster_label_start = {full_cluster_map[n]: label_start[n] for n in names}  # pure by construction
    cluster_label_end = {full_cluster_map[n]: label_end[n] for n in names}      # pure by construction

    return cluster_names, full_cluster_map, cluster_metric, cluster_weights, cluster_label_start, cluster_label_end
