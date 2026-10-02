import numpy as np
from src.banker import is_safe, need_matrix, try_grant

ALLOC = np.array([[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]])
MAX = np.array([[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]])


def test_need_matrix():
    assert need_matrix(MAX, ALLOC).tolist()[0] == [7, 4, 3]


def test_textbook_safe_state():
    safe, order = is_safe(np.array([3, 3, 2]), MAX, ALLOC)
    assert safe and order == [1, 3, 4, 0, 2]


def test_unsafe_state():
    safe, _ = is_safe(np.array([0, 0, 0]), MAX, ALLOC)
    assert not safe


def test_grant_safe_request():
    ok, avail, alloc = try_grant(1, np.array([1, 0, 2]), np.array([3, 3, 2]), MAX, ALLOC)
    assert ok and avail.tolist() == [2, 3, 0] and alloc[1].tolist() == [3, 0, 2]


def test_deny_unsafe_request():
    # Textbook follow-up: after P1's grant, P0 asking (0,2,0) would be unsafe.
    _, avail, alloc = try_grant(1, np.array([1, 0, 2]), np.array([3, 3, 2]), MAX, ALLOC)
    ok, _, _ = try_grant(0, np.array([0, 2, 0]), avail, MAX, alloc)
    assert not ok


def test_deny_request_above_need():
    ok, _, _ = try_grant(1, np.array([5, 0, 0]), np.array([9, 9, 9]), MAX, ALLOC)
    assert not ok
