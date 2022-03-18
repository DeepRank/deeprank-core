from deeprank_gnn.models.contact import AtomicContact, ResidueContact
from deeprank_gnn.models.structure import Atom, Residue, AtomicElement
from deeprank_gnn.domain.amino_acid import *


def test_atomic_contact():
    residue = Residue(None, 1, alanine)

    atom1 = Atom(residue, "CA", AtomicElement.C, (0.0, 0.0, 0.0), 1.0)
    atom2 = Atom(residue, "CB", AtomicElement.C, (0.0, 1.0, 0.0), 1.0)

    contact = AtomicContact(atom1, atom2, 1.0)
    contact.vanderwaals_potential = 1.0
    contact.coulomb_potential = 0.0

    assert contact.vanderwaals_potential == 1.0, "set vanderwaals value is lost"
    assert contact.coulomb_potential == 0.0, "set coulomb value is lost"
    assert contact.distance == 1.0, "set distance is lost"

    assert atom1 in contact
    assert atom2 in contact

    l = list(contact)
    assert len(l) == 2

    assert AtomicContact(atom1, atom2, 1.0) == AtomicContact(atom2, atom1, 1.0)


def test_residue_contact():
    residue1 = Residue(None, 1, alanine)
    residue2 = Residue(None, 2, glycine)

    contact = ResidueContact(residue1, residue2, 1.0)

    assert contact.distance == 1.0, "set distance is lost"

    assert residue1 in contact
    assert residue2 in contact

    l = list(contact)
    assert len(l) == 2

    assert ResidueContact(residue1, residue2, 1.0) == ResidueContact(residue2, residue1, 1.0)
