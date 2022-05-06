import numpy
from pdb2sql import pdb2sql

from deeprank_gnn.models.graph import Graph, Node
from deeprank_gnn.models.structure import Chain, Residue, Structure
from deeprank_gnn.feature.bsa import add_features
from deeprank_gnn.tools.graph import build_residue_graph, build_atomic_graph
from deeprank_gnn.tools.pdb import get_residue_contact_pairs
from deeprank_gnn.domain.feature import FEATURENAME_BURIEDSURFACEAREA


def _find_residue_node(graph, chain_id, residue_number):
    for node in graph.nodes:
        residue = node.id

        if residue.chain.id == chain_id and residue.number == residue_number:
            return node

    raise ValueError(f"Not found: {chain_id} {residue_number}")


def _find_atom_node(graph, chain_id, residue_number, atom_name):
    for node in graph.nodes:
        atom = node.id

        if atom.residue.chain.id == chain_id and atom.residue.number == residue_number and \
                atom.name == atom_name:

            return node

    raise ValueError(f"Not found: {chain_id} {residue_number} {atom_name}")


def test_add_features_residue():
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"

    residues = set([])
    for residue1, residue2 in get_residue_contact_pairs(pdb_path, "A", "B", 8.5):
        residues.add(residue1)
        residues.add(residue2)

    graph = build_residue_graph(residues, "1ATN-1w", 8.5)

    add_features(pdb_path, graph)

    # chain B ASP 93, at interface
    node = _find_residue_node(graph, "B", 93)

    assert node.features[FEATURENAME_BURIEDSURFACEAREA] > 0.0


def test_add_features_atom():
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"

    atoms = set([])
    for residue1, residue2 in get_residue_contact_pairs(pdb_path, "A", "B", 8.5):
        for atom in residue1.atoms:
            atoms.add(atom)
        for atom in residue2.atoms:
            atoms.add(atom)
    atoms = list(atoms)

    graph = build_atomic_graph(atoms, "1ATN-1w", 8.5)

    add_features(pdb_path, graph)

    # chain B ASP 93, at interface
    node = _find_atom_node(graph, "B", 93, "OD1")

    assert node.features[FEATURENAME_BURIEDSURFACEAREA] > 0.0
