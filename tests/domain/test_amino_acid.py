import numpy

from deeprank_gnn.domain.amino_acid import amino_acids


def test_all_different_onehot():

    codes = {}

    for amino_acid in amino_acids:
        for other in amino_acids:
            if other != amino_acid:
                assert not numpy.all(amino_acid.onehot == other.onehot), "{} is occupied by both {} and {}".format(amino_acid.onehot, other, amino_acid)
