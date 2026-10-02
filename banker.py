"""Banker's Algorithm: Need matrix, safety check and request check.

This module is the *ground truth* for the whole project. Every sequence and
every prefix is labelled safe / unsafe with `is_safe`.
"""
import numpy as np


def need_matrix(max_demand, allocation):
    """Need = Max - Allocation."""
    return np.asarray(max_demand) - np.asarray(allocation)


def is_safe(available, max_demand, allocation):
    """Safety algorithm.

    Returns (safe: bool, safe_sequence: list[int]).
    A state is safe if some order exists in which every process can get its
    remaining Need, finish, and release its allocation.
    """
    allocation = np.asarray(allocation)
    need = need_matrix(max_demand, allocation)
    work = np.asarray(available).copy()
    n = allocation.shape[0]
    finished = [False] * n
    order = []
    progress = True
    while progress:
        progress = False
        for i in range(n):
            if not finished[i] and np.all(need[i] <= work):
                work = work + allocation[i]          # process i finishes
                finished[i] = True
                order.append(i)
                progress = True
    return all(finished), order


def request_is_valid(i, request, available, max_demand, allocation):
    """Resource-request checks (request <= need, request <= available)."""
    need = need_matrix(max_demand, allocation)
    return bool(np.all(request <= need[i]) and np.all(request <= available))


def try_grant(i, request, available, max_demand, allocation):
    """Banker's resource-request algorithm.

    Pretend to grant, run the safety check, and only grant if the new state is
    safe. Returns (granted, new_available, new_allocation).
    """
    request = np.asarray(request)
    if not request_is_valid(i, request, available, max_demand, allocation):
        return False, available, allocation
    new_avail = available - request
    new_alloc = allocation.copy()
    new_alloc[i] = new_alloc[i] + request
    safe, _ = is_safe(new_avail, max_demand, new_alloc)
    if safe:
        return True, new_avail, new_alloc
    return False, available, allocation
