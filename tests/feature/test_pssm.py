from pdb2sql import pdb2sql
import numpy

from deeprank_gnn.domain.amino_acid import alanine
from deeprank_gnn.tools.pssm import parse_pssm
from deeprank_gnn.models.variant import SingleResidueVariant
from deeprank_gnn.feature.pssm import add_features
from deeprank_gnn.tools.graph import build_residue_graph, build_atomic_graph
from deeprank_gnn.tools.pdb import get_structure, get_surrounding_residues
from deeprank_gnn.domain.feature import (FEATURENAME_PSSM, FEATURENAME_PSSMDIFFERENCE,
                                         FEATURENAME_PSSMWILDTYPE, FEATURENAME_PSSMVARIANT,
                                         FEATURENAME_INFORMATIONCONTENT)


def test_add_features():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101m")
    finally:
        pdb._close()

    chain = structure.get_chain("A")
    with open("tests/data/pssm/101M/101M.A.pdb.pssm", 'rt') as f:
        chain.pssm = parse_pssm(f, chain)

    variant_residue = chain.residues[25]

    variant = SingleResidueVariant(variant_residue, alanine)

    residues = get_surrounding_residues(structure, variant_residue, 10.0)
    atoms = set([])
    for residue in residues:
        for atom in residue.atoms:
            atoms.add(atom)
    atoms = list(atoms)
    assert len(atoms) > 0

    graph = build_atomic_graph(atoms, "101M-25-atom", 4.5)
    add_features(pdb_path, graph, variant)

    for feature_name in (FEATURENAME_PSSM, FEATURENAME_PSSMDIFFERENCE,
                         FEATURENAME_PSSMWILDTYPE, FEATURENAME_PSSMVARIANT,
                         FEATURENAME_INFORMATIONCONTENT):
        assert numpy.any([node.features[feature_name] != 0.0
                          for node in graph.nodes])
