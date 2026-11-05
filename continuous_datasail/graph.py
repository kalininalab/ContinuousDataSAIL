"""The fold-transition graph and its decomposition into cycles.

Two splits of the same entities define a directed multigraph: the nodes are the
folds, and each entity that changes fold contributes one edge from where it
starts to where it ends. Walking from one split to the other means consuming
every edge exactly once.

Applying edges one at a time would break the split, because an entity leaving
``train`` has to be replaced by another arriving there if the fold sizes are to
hold. Decomposing the graph into *cycles* solves that: rotating the entities
around a closed walk moves every entity and leaves every fold the same size.
Whatever cannot be closed into a cycle is returned separately as *leftovers*,
and those do change fold sizes.
"""
from typing import Dict, List, Tuple

import networkx as nx

#: One interpolation step: a tuple of entity names. Length > 1 is a cycle, to be
#: rotated; length 1 is a leftover, to be sent straight to its end fold.
Operation = Tuple[str, ...]


def build_split_graph(
        start_split: Dict[str, str],
        end_split: Dict[str, str],
        names: List[str],
):
    """Build the directed multigraph of fold changes between two splits.

    Parameters
    ----------
    start_split : dict
        Entity name -> fold, at the start of the interpolation.
    end_split : dict
        Entity name -> fold, at the end.
    names : list of str
        Entities to consider. Both splits must label all of them.

    Returns
    -------
    networkx.MultiDiGraph
        Nodes are fold names. Every entity whose fold differs between the two
        splits adds one edge ``start -> end`` carrying the entity name in its
        ``name`` attribute. Entities that keep their fold add no edge, so they
        are never moved.
    """
    graph = nx.MultiDiGraph()
    graph.add_nodes_from({start_split[name] for name in names} | {end_split[name] for name in names})
    for name in names:
        u, v = start_split[name], end_split[name]
        if u != v:
            graph.add_edge(u, v, name=name)
    return graph


def decompose(graph: nx.MultiDiGraph):
    """Split the transition graph into cycles plus leftovers.

    The walk follows Hierholzer's algorithm for Eulerian circuits. Starting
    from a node, it consumes unused edges one by one and pushes the nodes it
    visits onto a stack. Revisiting a node on the stack closes a loop, which is
    popped off as a cycle; the walk then continues from where the loop closed.
    When a walk runs out of edges without closing, the entities collected so far
    cannot be rotated and become leftovers.

    A graph where every fold keeps its size has equal in- and out-degree at each
    node, so the walk can only ever stall at the node it began from, and that
    node closes a cycle instead of stalling. Such a graph therefore yields no
    leftovers at all. Unequal fold sizes between the endpoints are what produce
    them.

    The walk is deterministic for a given graph, but it is only one of many
    valid decompositions.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Output of :func:`build_split_graph`.

    Returns
    -------
    (list of Operation, list of str)
        The cycles, each a tuple of entity names in rotation order, and the
        leftover entity names.
    """
    # A plain adjacency list is faster to consume than the networkx view, and
    # popping from it is how edges get marked as used.
    adj: Dict[str, List[Tuple[str, str]]] = {n: [] for n in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        adj[u].append((v, data["name"]))

    cycles: List[Tuple[str, ...]] = []
    leftover: List[str] = []

    for start in list(graph.nodes()):
        while adj[start]:
            stack_nodes = [start]
            stack_names: List[str] = []
            pos = {start: 0}   # node -> its index in stack_nodes

            while True:
                node = stack_nodes[-1]
                if not adj[node]:
                    # Dead end: nothing here can be closed into a cycle.
                    leftover.extend(stack_names)
                    break
                nxt, name = adj[node].pop()
                stack_names.append(name)

                if nxt in pos:
                    # Closed a loop. Cut it off the stack and keep walking.
                    idx = pos[nxt]
                    cycles.append(tuple(stack_names[idx:]))
                    del stack_names[idx:]
                    for removed in stack_nodes[idx + 1:]:
                        del pos[removed]
                    del stack_nodes[idx + 1:]
                else:
                    stack_nodes.append(nxt)
                    pos[nxt] = len(stack_nodes) - 1

    return cycles, leftover
