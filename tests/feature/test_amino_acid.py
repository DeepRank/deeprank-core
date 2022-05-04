from pdb2sql import pdb2sql

from deeprank_gnn.domain.amino_acid import serine
from deeprank_gnn.models.variant import SingleResidueVariant
from deeprank_gnn.models.graph import Graph, Node
from deeprank_gnn.models.structure import Chain, Residue
from deeprank_gnn.feature.sasa import add_features
from deeprank_gnn.tools.graph import build_residue_graph, build_atomic_graph
from deeprank_gnn.tools.pdb import get_structure, get_surrounding_residues
from deeprank_gnn.domain.feature import FEATURENAME_HYDROGENBONDDONORSDIFFERENCE, FEATURENAME_SIZEDIFFERENCE
from deeprank_gnn.feature.amino_acid import add_features


def test_add_features():
    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()

    residue = structure.chains[0].residues[25]
    variant = SingleResidueVariant(residue, serine)  # GLY -> SER

    residues = get_surrounding_residues(structure, residue, 10.0)
    assert len(residues) > 0

    graph = build_residue_graph(residues, "101m-25", 4.5)

    add_features(pdb_path, graph, variant)

    for node in graph.nodes:
        if node.id == variant.residue:  # GLY -> SER
            assert node.features[FEATURENAME_SIZEDIFFERENCE] > 0
            assert node.features[FEATURENAME_HYDROGENBONDDONORSDIFFERENCE] > 0
