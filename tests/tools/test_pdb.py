import tempfile
import os

import numpy
from pdb2sql import pdb2sql
from deeprank_gnn.domain.amino_acid import alanine, valine
from deeprank_gnn.models.structure import AtomicElement
from tests.help import memory_limit
from deeprank_gnn.tools.pdb import (
    get_residue_contact_pairs,
    get_surrounding_residues,
    add_hydrogens,
)

def test_get_structure_complete():
    pdb_path = "tests/data/pdb/101M/101M.pdb"

    residues = get_surrounding_residues(pdb_path, "A", 25, None, 30.0)
    chain = residues[0].chain
    assert chain.id == "A"

    assert len(chain.residues) == 154
    residue = chain.residues[1]
    assert residue.number == 1
    assert residue.chain == chain
    assert residue.amino_acid == valine

    assert len(residue.atoms) == 7
    atom = residue.atoms[1]
    assert atom.name == "CA"
    assert atom.position[0] == 27.263  # x coord from PDB file
    assert atom.element == AtomicElement.C
    assert atom.residue == residue


def test_get_structure_from_nmr_with_dna():
    pdb_path = "tests/data/pdb/1A6B/1A6B.pdb"

    residues = get_surrounding_residues(pdb_path, "B", 14, None, 30.0)
    assert any(residue.amino_acid is None for residue in residues)  # DNA


def test_residue_contact_pairs():

    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb"

    residue_pairs = get_residue_contact_pairs(pdb_path, "A", "B", 8.5)

    assert len(residue_pairs) > 0

    for residue1, residue2 in residue_pairs:
        assert residue1 != residue2, f"residue {residue1} was paired with itself"


def test_surrounding_residues():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    # A nicely centered residue
    chain_id = "A"
    residue_number = 138

    close_residues = get_surrounding_residues(pdb_path, chain_id, residue_number, None, 10.0)

    assert len(close_residues) > 0, "no close residues found"
    assert len(close_residues) < 1000, "all residues were picked"
    assert any([residue.number == residue_number and residue.chain.id == chain_id
                for residue in close_residues]), "the centering residue wasn't included"


@memory_limit(1024 * 1024 * 1024)
def test_surrounding_residues_large_structure():

    pdb_path = "tests/data/pdb/2Y69/2Y69.pdb"
    chain_id = "E"
    residue_number = 74

    close_residues = get_surrounding_residues(pdb_path, chain_id, residue_number, None, 10.0)

    assert len(close_residues) > 0, "no close residues found"
    assert len(close_residues) < 1000, "too many residues were picked"
    assert any([residue.number == residue_number and residue.chain.id == chain_id
                for residue in close_residues]), "the centering residue wasn't included"


def test_hydrogens():
    test_file, test_path = tempfile.mkstemp()
    os.close(test_file)

    try:
        add_hydrogens("tests/data/pdb/2Y69/2Y69.pdb", test_path)

        assert os.path.isfile(test_path)
        with open(test_path, 'rt') as f:
            assert any(["PHE P  94" in line for line in f])
    finally:
        os.remove(test_path)
