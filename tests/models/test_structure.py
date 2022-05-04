import pickle

from pdb2sql import pdb2sql

from deeprank_gnn.models.structure import AtomicElement
from deeprank_gnn.tools.pdb import get_structure


def test_element():
    value = AtomicElement.C.onehot

