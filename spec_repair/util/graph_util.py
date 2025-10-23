from collections import deque
from copy import deepcopy
from typing import Set, List, Dict

import networkx as nx


def merge_sets(sets: Set[frozenset[any]]) -> List[Set[any]]:
    merged_sets = []

    while sets:
        cur_set: Set[any] = set(sets.pop())
        sets_copy = deepcopy(sets)

        for s in sets_copy:
            if not cur_set.isdisjoint(s):
                cur_set |= s
                sets.remove(s)

        if sets == sets_copy:
            merged_sets.append(cur_set)
        else:
            sets.add(frozenset(cur_set))

    return merged_sets


def remove_reflexive_relations(graph: nx.DiGraph):
    """
    e.g.: [(A->A), (B->C), (B->B), (A->C)] => [(B->C), (A->C)]
    :param graph: directed graph
    :return: same directed graph, without reflexive edges
    """
    for u in graph.nodes():
        if (u, u) in graph.edges():
            graph.remove_edge(u, u)


def merge_on_bidirectional_edges(graph: nx.DiGraph):
    """
    Edges that point to each other will be removed, and nodes will be merged.
    e.g.: [(A->B), (B->A), (B->C), (C->D), (D->C)] => [((A,B)->(C,D))]
    :param graph: non-reflexive, non-transitive directed graph
    :return: modified directed graph, as described above
    """
    edges_to_merge: Set[frozenset[str]] = get_edges_to_merge(graph)
    merged_nodes_set: List[Set[str]] = merge_sets(edges_to_merge)
    node_mapping: Dict[str, str] = dict()
    for merged_nodes in merged_nodes_set:
        new_name = ','.join(sorted(merged_nodes))
        for node in merged_nodes:
            node_mapping[node] = new_name
    nx.relabel_nodes(graph, node_mapping, copy=False)


def get_edges_to_merge(graph: nx.DiGraph) -> Set[frozenset[str]]:
    """
    :param graph: directed graph
    :return: set of sets of the nodes representing each bidirectional edge
    """
    pairs_to_merge = set()
    for u, v in graph.edges():
        if graph.has_edge(v, u):
            edges = [u, v]
            edges.sort()
            pairs_to_merge.add(frozenset(edges))
    return pairs_to_merge


def remove_transitive_relations(graph: nx.DiGraph, root_node: str):
    """
    e.g.: [(A->B), (B->C), (A->C)] => [(A->B), (B->C)]
    :param graph: directed graph with no reflexive relations
    :param root_node: starting node for direct transitive relations to be considered
    :return: same directed graph, without edges for transitivity
    """
    nodes_to_check = deque()
    visited_nodes = set()
    nodes_to_check.append(root_node)
    while len(nodes_to_check) > 0:
        # Find transitive edges and remove them from the copy
        cur_node = nodes_to_check.popleft()
        visited_nodes.add(cur_node)
        prev_graph = graph.copy()
        for u in prev_graph.successors(cur_node):
            if u in graph.successors(cur_node):
                for v in prev_graph.successors(u):
                    if (cur_node, v) in graph.edges() and u != v:
                        graph.remove_edge(cur_node, v)

        for u in prev_graph.predecessors(cur_node):
            if u in graph.predecessors(cur_node):
                for v in prev_graph.predecessors(u):
                    if (v, cur_node) in graph.edges() and u != v:
                        graph.remove_edge(v, cur_node)
        for u in graph.successors(cur_node):
            if u not in visited_nodes:
                nodes_to_check.append(u)
