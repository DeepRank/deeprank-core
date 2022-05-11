from pdb2sql import pdb2sql

from deeprank_gnn.domain.forcefield import atomic_forcefield
from deeprank_gnn.tools.pdb import get_surrounding_residues
from deeprank_gnn.domain.amino_acid import arginine, glutamate


def test_atomic_forcefield():

    residues = get_surrounding_residues("tests/data/pdb/101M/101M.pdb",
                                        "A", 40, None, 100.0)

    # The arginine C-zeta should get a positive charge
    arg = [r for r in residues if r.amino_acid == arginine][0]
    cz = [a for a in arg.atoms if a.name == "CZ"][0]
    assert atomic_forcefield.get_charge(cz) == 0.640

    # The glutamate O-epsilon should get a negative charge
    glu = [r for r in residues if r.amino_acid == glutamate][0]
    oe2 = [a for a in glu.atoms if a.name == "OE2"][0]
    assert atomic_forcefield.get_charge(oe2) == -0.800

    # The forcefield should treat terminal oxygen differently
    oxt = [a for a in residues[0].chain.model.get_atoms() if a.name == "OXT"][0]
    o = [a for a in oxt.residue.atoms if a.name == "O"][0]
    assert atomic_forcefield.get_charge(oxt) == -0.800
    assert atomic_forcefield.get_charge(o) == -0.800
