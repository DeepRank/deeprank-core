import tempfile
import logging
import os
from time import time
from typing import Optional, List
import subprocess

from scipy.spatial import distance_matrix
import numpy
from pdb2sql import interface as get_interface
from pdb2sql import pdb2sql

from deeprank_gnn.models.structure import Atom, Residue, Chain, Structure, AtomicElement
from deeprank_gnn.domain.amino_acid import amino_acids
from deeprank_gnn.models.pair import Pair
from deeprank_gnn.domain.forcefield import atomic_forcefield
from deeprank_gnn.models.contact import ResidueContact, AtomicContact
from deeprank_gnn.feature.atomic_contact import get_coulomb_potentials, get_lennard_jones_potentials


_log = logging.getLogger(__name__)


def is_xray(pdb_file):
    "check that an open pdb file is an x-ray structure"

    for line in pdb_file:
        if line.startswith("EXPDTA") and "X-RAY DIFFRACTION" in line:
            return True

    return False


def add_hydrogens(input_pdb_path, output_pdb_path):
    "this requires reduce: https://github.com/rlabduke/reduce"

    if not os.path.isfile(input_pdb_path):
        raise FileNotFoundError(input_pdb_path)

    tmp_file, tmp_path = tempfile.mkstemp()
    os.close(tmp_file)

    with open(tmp_path, 'wb') as f:
        subprocess.run(["reduce", "-Quiet", input_pdb_path], stdout=f, check=True)

    try:
        with open(tmp_path, 'rt') as f:
            with open(output_pdb_path, 'wt') as g:
                for line in f:
                    g.write(line.replace("   new", "").replace("   std",""))
    finally:
        os.remove(tmp_path)


def _add_atom_to_residue(atom, residue):

    for other_atom in residue.atoms:
        if other_atom.name == atom.name:
            # Don't allow two atoms with the same name, pick the highest
            # occupancy
            if other_atom.occupancy < atom.occupancy:
                other_atom.change_altloc(atom)

            return other_atom

    # not there yet, add it
    residue.add_atom(atom)
    return atom


def _to_atoms(atom_rows: numpy.ndarray, structure: Structure) -> List[Atom]:
    """
        Args:
            atom_rows: output from pdb2sql
            structure: to add the atoms to
    """

    elements_by_symbol = {element.name: element for element in AtomicElement}
    amino_acids_by_code = {
        amino_acid.three_letter_code: amino_acid for amino_acid in amino_acids
    }

    # We need these intermediary dicts to keep track of which residues and
    # chains have already been created.
    atoms = set([])

    # Iterate over the atom output from pdb2sql
    for row in atom_rows:
        (
            x,
            y,
            z,
            atom_name,
            altloc,
            occupancy,
            element,
            chain_id,
            residue_number,
            residue_name,
            insertion_code,
        ) = row

        # Make sure not to take the same atom twice.
        if altloc is not None and altloc != "" and altloc != "A":
            continue

        # We use None to indicate that the residue has no insertion code.
        if insertion_code == "":
            insertion_code = None

        # The amino acid is only valid when we deal with protein residues.
        if residue_name in amino_acids_by_code:
            amino_acid = amino_acids_by_code[residue_name]
        else:
            amino_acid = None

        # Turn the x,y,z into a vector:
        atom_position = numpy.array([x, y, z])

        # Init chain.
        if structure.has_chain(chain_id):
            chain = structure.get_chain(chain_id)
        else:
            chain = Chain(structure, chain_id)
            structure.add_chain(chain)

        # Init residue.
        for chain_residue in chain.residues:
            if chain_residue.number == residue_number and \
               chain_residue.insertion_code == insertion_code:

                atom_residue = chain_residue
                break
        else:
            atom_residue = Residue(chain, residue_number, amino_acid, insertion_code)
            chain.add_residue(atom_residue)

        # Init atom.
        atom = Atom(
            atom_residue, atom_name, elements_by_symbol[element], atom_position, occupancy
        )
        atom = _add_atom_to_residue(atom, atom_residue)
        atoms.add(atom)

    return list(atoms)


def get_residue_contact_pairs( # pylint: disable=too-many-locals
    pdb_path: str,
    chain_id1: str,
    chain_id2: str,
    distance_cutoff: float,
) -> List[Pair]:

    """Get the residues that contact each other at a protein-protein interface.

    Args:
        pdb_path: the path of the pdb file, that the structure was built from
        chain_id1: first protein chain identifier
        chain_id2: second protein chain identifier
        distance_cutoff: max distance between two interacting residues

    Returns: the pairs of contacting residues
    """

    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    structure_id = f"interface-{pdb_name}-{chain_id1}:{chain_id2}"

    # Find out which residues are pairs
    interface = get_interface(pdb_path)
    pdb = pdb2sql(pdb_path)
    try:
        contact_residues = interface.get_contact_residues(
            cutoff=distance_cutoff,
            chain1=chain_id1,
            chain2=chain_id2,
            return_contact_pairs=True,
        )
    finally:
        interface._close() # pylint: disable=protected-access

    # Map to residue objects
    structure = Structure(structure_id)
    residue_pairs = set([])
    for (residue_chain_id1, residue_number1, residue_name1), residue_keys2 in contact_residues.items():

        residue1_atom_rows = pdb.get("x,y,z,name,altLoc,occ,element,chainID,resSeq,resName,iCode",
                                     model=0,
                                     chainID=residue_chain_id1,
                                     resSeq=residue_number1,
                                     resName=residue_name1)

        if len(residue1_atom_rows) == 0:
            raise ValueError(
                f"Not found: {pdb_path} {residue_chain_id1} {residue_number1} {residue_name1}"
            )

        residue1_atoms = _to_atoms(residue1_atom_rows, structure)
        residue1 = residue1_atoms[0].residue

        for residue_chain_id2, residue_number2, residue_name2 in residue_keys2:

            residue2_atom_rows = pdb.get("x,y,z,name,altLoc,occ,element,chainID,resSeq,resName,iCode",
                                         model=0,
                                         chainID=residue_chain_id2,
                                         resSeq=residue_number2,
                                         resName=residue_name2)

            if len(residue1_atom_rows) == 0:
                raise ValueError(
                    f"Not found: {pdb_path} {residue_chain_id2} {residue_number2} {residue_name2}"
                )

            residue2_atoms = _to_atoms(residue2_atom_rows, structure)
            residue2 = residue2_atoms[0].residue

            residue_pairs.add(Pair(residue1, residue2))

    return list(residue_pairs)


def get_surrounding_residues(pdb_path: str, chain_id: str,
                             residue_number: int, insertion_code: Optional[str],
                             radius: float) -> List[Residue]:

    """ Get the residues that lie within a radius around a residue.

        Args:
            pdb_path: points to PDB file
            chain_id: identifies the residue in the pdb
            residue_number: identifies the residue in the pdb
            insertion_code: identifies the residue in the pdb
            radius: max distance in Ångström between atoms of the residue and the other residues
    """

    pdb = pdb2sql(pdb_path)

    structure_atom_rows = pdb.get("x,y,z,name,altLoc,occ,element,chainID,resSeq,resName,iCode", model=0)
    structure_atom_positions = [row[:3] for row in structure_atom_rows]

    # convert insertion code to pdb2sql's format
    if insertion_code is None:
        insertion_code = ""

    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    structure_id = f"{pdb_name}-centered-{chain_id}:{residue_number}{insertion_code}"

    residue_atom_positions = pdb.get("x,y,z",
                                     chainID=chain_id,
                                     resSeq=residue_number, iCode=insertion_code,
                                     model=0)

    if len(residue_atom_positions) == 0:
        raise ValueError(
            f"Not found: {pdb_path} {chain_id} {residue_number}{insertion_code}"
        )

    distances = distance_matrix(structure_atom_positions, residue_atom_positions, p=2)
    neighbours = distances < radius

    neighbour_indices = numpy.nonzero(neighbours)[0]

    neighbour_atom_rows = [structure_atom_rows[i] for i in neighbour_indices]

    structure = Structure(structure_id)
    atoms = _to_atoms(neighbour_atom_rows, structure)

    residues = set([atom.residue for atom in atoms])

    return list(residues)


