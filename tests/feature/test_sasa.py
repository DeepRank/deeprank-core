import numpy
from pdb2sql import pdb2sql

from deeprank_gnn.domain.amino_acid import alanine
from deeprank_gnn.models.variant import SingleResidueVariant
from deeprank_gnn.models.graph import Graph, Node
from deeprank_gnn.models.structure import Chain, Residue
from deeprank_gnn.feature.sasa import add_features
from deeprank_gnn.tools.graph import build_residue_graph, build_atomic_graph
from deeprank_gnn.tools.pdb import get_structure, get_surrounding_residues
from deeprank_gnn.domain.feature import FEATURENAME_SASA


def _get_residue(chain: Chain, number: int) -> Residue:
    for residue in chain.residues:
        if residue.number == number:
            return residue

    raise ValueError(f"Not found: {number}")


def _get_residue_node(graph: Graph, number: int) -> Node:
    for node in graph.nodes:
        residue = node.id
        if residue.number == number:
            return node

    raise ValueError(f"Not found: {number}")


def _get_atom_node(graph: Graph, residue_number: int, atom_name: str) -> Node:
    for node in graph.nodes:
        atom = node.id
        if atom.residue.number == residue_number and atom.name == atom_name:
            return node

    raise ValueError(f"Not found: {residue_number} {atom_name}")


def test_add_features_to_residues():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()

    residue = _get_residue(structure.chains[0], 108)
    variant = SingleResidueVariant(residue, alanine)

    residues = get_surrounding_residues(structure, residue, 10.0)
    assert len(residues) > 0

    graph = build_residue_graph(residues, "101M-108-res", 4.5)
    add_features(pdb_path, graph, variant)

    # check for NaN
    assert not any([numpy.isnan(node.features[FEATURENAME_SASA])
                    for node in graph.nodes])

    # surface residues should have large area
    surface_residue_node = _get_residue_node(graph, 105)
    assert surface_residue_node.features[FEATURENAME_SASA] > 25.0

    # buried residues should have small area
    buried_residue_node = _get_residue_node(graph, 72)
    assert buried_residue_node.features[FEATURENAME_SASA] < 25.0, buried_residue_node.features[FEATURENAME_SASA]


def test_add_features_to_atoms():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()

    residue = _get_residue(structure.chains[0], 108)
    variant = SingleResidueVariant(residue, alanine)

    residues = get_surrounding_residues(structure, residue, 10.0)
    atoms = set([])
    for residue in residues:
        for atom in residue.atoms:
            atoms.add(atom)
    atoms = list(atoms)
    assert len(atoms) > 0

    graph = build_atomic_graph(atoms, "101M-108-atom", 4.5)
    add_features(pdb_path, graph, variant)

    # check for NaN
    assert not any([numpy.isnan(node.features[FEATURENAME_SASA])
                    for node in graph.nodes])

    # surface atoms should have large area
    surface_atom_node = _get_atom_node(graph, 105, "OE2")
    assert surface_atom_node.features[FEATURENAME_SASA] > 25.0

    # buried atoms should have small area
    buried_atom_node = _get_atom_node(graph, 72, "CG")
    assert buried_atom_node.features[FEATURENAME_SASA] == 0.0, buried_atom_node.features[FEATURENAME_SASA]
