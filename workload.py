"""Synthetic workload generator.

A *sequence* is a list of T resource requests granted one after another
(without Banker's blocking them) so that we can watch how the state evolves.
Every prefix state is labelled with the Banker's safety check.

Workloads
---------
RANDOM       : random valid requests (used for training, and as a no-shift test)
LATE_UNSAFE  : first half only picks requests that keep the state safe, second
               half switches to requests that push the state into unsafe
               territory (the deliberate mid-trace shift).
"""
import numpy as np
from .banker import is_safe


DEMAND_LOAD = 2.0   # total claimed demand ~ DEMAND_LOAD x capacity (tunable)


def make_system(rng, n_proc, n_res, load=None):
    load = DEMAND_LOAD if load is None else load
    total = rng.integers(8, 16, size=n_res)
    max_demand = np.zeros((n_proc, n_res), dtype=int)
    for j in range(n_res):
        hi = max(1, int(round(2 * load * total[j] / n_proc)))
        max_demand[:, j] = np.minimum(rng.integers(0, hi + 1, size=n_proc), total[j])
    return total, max_demand


def _candidate(rng, alloc, max_demand, avail, greedy=False):
    n = alloc.shape[0]
    need = max_demand - alloc
    for _ in range(10):
        i = int(rng.integers(n))
        cap = np.minimum(need[i], avail)
        if cap.sum() == 0:
            continue
        if greedy:                       # take a large share of what is allowed
            req = np.floor(cap * rng.uniform(0.5, 0.9)).astype(int)   # big but partial (full need would let it finish)
        else:
            req = np.array([rng.integers(0, c + 1) for c in cap])
        if req.sum() == 0:
            req[int(np.argmax(cap))] = 1
        return i, req
    return None


def generate_sequence(rng, n_proc, n_res, kind="RANDOM", length=None, tries=60):
    length = length or 4 * n_proc
    total, max_demand = make_system(rng, n_proc, n_res)
    alloc = np.zeros((n_proc, n_res), dtype=int)
    avail = total.copy()
    shift = length // 2
    steps, safe_flags = [], []
    for t in range(length):
        want_unsafe = (kind == "LATE_UNSAFE" and t >= shift)
        want_safe = (kind == "LATE_UNSAFE" and t < shift)
        chosen = None
        for _ in range(tries):
            cand = _candidate(rng, alloc, max_demand, avail, greedy=(kind == "LATE_UNSAFE" and rng.random() < 0.6))
            if cand is None:
                break
            i, req = cand
            if kind == "RANDOM":
                chosen = cand
                break
            a2 = alloc.copy()
            a2[i] += req
            s, _ = is_safe(avail - req, max_demand, a2)
            if want_safe and s:
                chosen = cand
                break
            if want_unsafe and not s:
                chosen = cand
                break
            if want_unsafe and chosen is None:
                chosen = cand            # fallback: any valid request (shift phase only)
        if chosen is None:               # nothing can be requested: no-op step
            steps.append((0, np.zeros(n_res, dtype=int)))
        else:
            i, req = chosen
            alloc[i] += req
            avail = avail - req
            steps.append((i, req))
        safe, _ = is_safe(avail, max_demand, alloc)
        safe_flags.append(safe)
    return {
        "kind": kind, "total": total, "max": max_demand, "steps": steps,
        "safe_flags": safe_flags, "shift_step": shift if kind == "LATE_UNSAFE" else None,
    }


def replay_states(seq):
    """Yield (alloc, avail) after each step of the sequence."""
    alloc = np.zeros_like(seq["max"])
    avail = seq["total"].copy()
    for i, req in seq["steps"]:
        alloc[i] += req
        avail = avail - req
        yield alloc.copy(), avail.copy()


def label(seq):
    """Sequence label (True = unsafe at some point) and first unsafe step."""
    flags = seq["safe_flags"]
    if all(flags):
        return False, None
    return True, flags.index(False)
