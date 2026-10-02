"""Resource Allocation Graph (RAG) cycle detection using DFS.

Nodes are ("P", i) for processes and ("R", j) for resources.
  request edge   P -> R : process waits for resource
  assignment edge R -> P : resource is held by process
For single-instance resources a cycle means deadlock. For multi-instance
resources a cycle is only a necessary condition, so the Banker's algorithm
stays the ground truth in the experiments.
"""


def build_rag(assignments, requests):
    """assignments: list of (resource, process); requests: list of (process, resource)."""
    graph = {}
    for r, p in assignments:
        graph.setdefault(("R", r), []).append(("P", p))
        graph.setdefault(("P", p), [])
    for p, r in requests:
        graph.setdefault(("P", p), []).append(("R", r))
        graph.setdefault(("R", r), [])
    return graph


def has_cycle(graph):
    """Iterative-safe DFS cycle detection (white/grey/black colouring)."""
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {v: WHITE for v in graph}

    def dfs(u):
        colour[u] = GREY
        for v in graph[u]:
            if colour[v] == GREY:
                return True
            if colour[v] == WHITE and dfs(v):
                return True
        colour[u] = BLACK
        return False

    return any(colour[v] == WHITE and dfs(v) for v in list(graph))
