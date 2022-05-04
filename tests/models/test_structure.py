import pickle

from pdb2sql import pdb2sql

from deeprank_gnn.models.structure import AtomicElement


def test_element():
    value = AtomicElement.C.onehot

