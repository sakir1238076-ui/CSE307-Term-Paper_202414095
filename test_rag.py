from src.rag import build_rag, has_cycle


def test_cycle_is_deadlock():
    # R0 held by P0, R1 held by P1; P0 wants R1, P1 wants R0
    g = build_rag([(0, 0), (1, 1)], [(0, 1), (1, 0)])
    assert has_cycle(g)


def test_no_cycle():
    g = build_rag([(0, 0)], [(1, 0)])
    assert not has_cycle(g)
