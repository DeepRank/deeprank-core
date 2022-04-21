import numpy

from pdb2sql import pdb2sql
from deeprank_gnn.tools.pdb import get_residue_contact_pairs, get_surrounding_residues, find_neighbour_atoms
from deeprank_gnn.domain.amino_acid import valine
from deeprank_gnn.models.structure import AtomicElement
from tests.help import memory_limit


def test_get_structure_complete():
    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()

    assert structure is not None

    assert len(structure.chains) == 1
    chain = structure.chains[0]
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

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()

    assert structure is not None
    assert structure.chains[0].residues[0].amino_acid is None  # DNA


def test_residue_contact_pairs():

    residue_pairs = get_residue_contact_pairs("tests/data/pdb/1ATN/1ATN_1w.pdb", "1ATN", "A", "B", 8.5)

    assert len(residue_pairs) > 0


def test_surrounding_residues():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    # A nicely centered residue
    close_residues = get_surrounding_residues(pdb_path, 'A', 138, None, 10.0)

    assert len(close_residues) > 0, "no close residues found"
    assert len(close_residues) < 200, "too many residues were picked"
    assert any([residue.number == 138 for residue in close_residues]), "the centering residue wasn't included"


@memory_limit(1024 * 1024 * 1024)
def test_surrounding_residues_large_structure():

    pdb_path = "tests/data/pdb/2Y69/2Y69.pdb"

    close_residues = get_surrounding_residues(pdb_path, 'E', 74, None, 10.0)

    assert len(close_residues) > 0, "no close residues found"
    assert len(close_residues) < 200, "too many residues found"
    assert any([residue.number == 74 and residue.chain.id == 'E' for residue in close_residues]), "the centering residue wasn't included"


def test_neighbour_atoms():

    pdb_path = "tests/data/pdb/101M/101M.pdb"

    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, "101M")
    finally:
        pdb._close()

    atoms = structure.get_atoms()

    atom_pairs = find_neighbour_atoms(atoms, 4.5)

    assert len(atom_pairs) > 0, "no atom pairs found"
    assert len(atom_pairs) < numpy.square(len(atoms)), "every two atoms were paired"

    for atom1, atom2 in atom_pairs:
        assert atom1 != atom2, "atom {} was paired with itself".format(atom1)
