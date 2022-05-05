from pdb2sql import pdb2sql

from deeprank_gnn.domain.amino_acid import serine, glycine
from deeprank_gnn.models.variant import SingleResidueVariant
from deeprank_gnn.models.graph import Graph, Node
from deeprank_gnn.models.structure import Chain, Residue
from deeprank_gnn.tools.graph import build_residue_graph, build_atomic_graph
from deeprank_gnn.tools.pdb import get_surrounding_residues
from deeprank_gnn.domain.feature import FEATURENAME_HYDROGENBONDDONORSDIFFERENCE, FEATURENAME_SIZEDIFFERENCE
from deeprank_gnn.feature.amino_acid import add_features


def test_add_features():
    pdb_path = "tests/data/pdb/101M/101M.pdb"

    residues = get_surrounding_residues(pdb_path, "A", 25, None, 10.0)
    assert len(residues) > 0

    for residue in residues:
        if residue.amino_acid == glycine:
            variant_residue = residue

    variant = SingleResidueVariant(variant_residue, serine)  # GLY -> SER

    graph = build_residue_graph(residues, "101m-25", 4.5)

    add_features(pdb_path, graph, variant)

    matching = 0
    for node in graph.nodes:
        if node.id == variant_residue:
            assert node.features[FEATURENAME_SIZEDIFFERENCE] > 0
            assert node.features[FEATURENAME_HYDROGENBONDDONORSDIFFERENCE] > 0
            matching += 1

    assert matching == 1
