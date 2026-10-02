"""Early-risk classifier (Decision Tree) using only features of a partial prefix."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .workload import replay_states, label

FEATURE_NAMES = ["utilization", "min_avail_frac", "request_pressure", "max_resource_pressure"]


def extract_features(alloc, avail, max_demand, total):
    """Simple aggregate features computed from the state *so far* only."""
    need = max_demand - alloc
    util = alloc.sum() / total.sum()
    min_avail = float(np.min(avail / total))
    pressure = need.sum() / (avail.sum() + 1.0)
    res_pressure = float(np.max(need.sum(axis=0) / (avail + 1.0)))
    return [util, min_avail, pressure, res_pressure]


def sequence_features(seq):
    return np.array([extract_features(a, v, seq["max"], seq["total"]) for a, v in replay_states(seq)])


def build_dataset(seqs):
    X, y = [], []
    for s in seqs:
        unsafe, _ = label(s)
        f = sequence_features(s)
        X.append(f)
        y.extend([int(unsafe)] * len(f))
    return np.vstack(X), np.array(y)


def train(seqs, max_depth=4, seed=0):
    X, y = build_dataset(seqs)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    clf.fit(X, y)
    return clf


def risk_curve(clf, seq):
    """P(risky) after every step of the sequence."""
    f = sequence_features(seq)
    if 1 not in clf.classes_:
        return np.zeros(len(f))
    return clf.predict_proba(f)[:, list(clf.classes_).index(1)]


def explain(features, prob):
    """Rule-based template explanation (bonus track)."""
    util, min_av, pres, rpres = features
    flag = "RISK FLAGGED" if prob >= 0.5 else "No risk flagged"
    return (f"{flag} (risk={prob:.0%}): utilization is {util:.0%}, scarcest resource has "
            f"{min_av:.0%} free, request pressure is {pres:.2f}.")
