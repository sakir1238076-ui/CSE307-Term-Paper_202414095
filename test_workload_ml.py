import numpy as np
from src.workload import generate_sequence, label
from src.ml import sequence_features


def test_late_unsafe_is_safe_before_shift():
    rng = np.random.default_rng(0)
    for _ in range(40):
        s = generate_sequence(rng, 6, 3, "LATE_UNSAFE")
        assert all(s["safe_flags"][: s["shift_step"]])


def test_unsafe_is_permanent():
    rng = np.random.default_rng(1)
    for _ in range(40):
        s = generate_sequence(rng, 6, 3, "RANDOM")
        unsafe, first = label(s)
        if unsafe:
            assert not any(s["safe_flags"][first:])


def test_features_use_prefix_only():
    """No leakage: features at step t must not change when later steps are removed."""
    rng = np.random.default_rng(2)
    s = generate_sequence(rng, 6, 3, "LATE_UNSAFE")
    t = 7
    cut = dict(s, steps=s["steps"][:t], safe_flags=s["safe_flags"][:t])
    assert np.allclose(sequence_features(cut), sequence_features(s)[:t])
