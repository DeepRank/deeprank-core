import numpy

from deeprank_gnn.domain.amino_acid import amino_acids


def test_all_different_onehot():

    for amino_acid in amino_acids:
        for other in amino_acids:
            if other != amino_acid:
                assert not numpy.all(
                    amino_acid.onehot == other.onehot
                ), f"{amino_acid.onehot} is occupied by both {other} and {amino_acid}"
