from deeprank_gnn.models.pair import Pair


def test_order_independency():
    # These should be the same:
    pair1 = Pair(1, 2)
    pair2 = Pair(2, 1)

    # test comparing:
    assert pair1 == pair2

    # test hashing:
    d = {pair1: 1}
    d[pair2] = 2
    assert d[pair1] == 2


def test_uniqueness():
    # These should be different:
    pair1 = Pair(1, 2)
    pair2 = Pair(1, 3)

    # test comparing:
    assert pair1 != pair2

    # test hashing:
    d = {pair1: 1}
    d[pair2] = 2
    assert d[pair1] == 1



