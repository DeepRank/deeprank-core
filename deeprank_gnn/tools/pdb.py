from time import time
from typing import Optional, List
import logging
import subprocess

from scipy.spatial import distance_matrix
import numpy
import torch
from pdb2sql import pdb2sql, interface as get_interface

from deeprank_gnn.models.structure import Atom, Residue, Chain, Structure, AtomicElement
from deeprank_gnn.domain.amino_acid import amino_acids
from deeprank_gnn.models.pair import Pair


_log = logging.getLogger(__name__)


def is_xray(pdb_file):
    "check that an open pdb file is an x-ray structure"

    for line in pdb_file:
        if line.startswith("EXPDTA") and "X-RAY DIFFRACTION" in line:
            return True

    return False


def add_hydrogens(input_pdb_path, output_pdb_path):
    "this requires reduce: https://github.com/rlabduke/reduce"

    with open(output_pdb_path, 'wt') as f:
        p = subprocess.run(["reduce", input_pdb_path], stdout=subprocess.PIPE)
        for line in p.stdout.decode().split('\n'):
            f.write(line.replace("   new", "") + "\n")


def _add_atom_to_residue(atom, residue):

    for other_atom in residue.atoms:
        if other_atom.name == atom.name:
            # Don't allow two atoms with the same name, pick the highest occupancy
            if other_atom.occupancy < atom.occupancy:
                other_atom.change_altloc(atom)
                return other_atom

    # not there yet, add it
    residue.add_atom(atom)
    return atom


def _to_atoms(atom_rows: numpy.ndarray, structure_id: str) -> List[Atom]:
    """
        Args:
            atom_rows: output from pdb2sql
            id: unique id for the pdb structure
    """

    amino_acids_by_code = {amino_acid.three_letter_code: amino_acid for amino_acid in amino_acids}
    elements_by_symbol = {element.name: element for element in AtomicElement}

    # We need these intermediary dicts to keep track of which residues and chains have already been created.
    chains = {}
    residues = {}
    atoms = set([])

    structure = Structure(structure_id)

    # Iterate over the atom output from pdb2sql
    for x, y, z, atom_number, atom_name, alt_loc, occupancy, element_symbol, chain_id, residue_number, residue_name, insertion_code in atom_rows:

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
        if chain_id not in chains:

            chain = Chain(structure, chain_id)
            chains[chain_id] = chain
            structure.add_chain(chain)
        else:
            chain = chains[chain_id]

        # Init residue.
        residue_id = (chain_id, residue_number, insertion_code)
        if residue_id not in residues:

            residue = Residue(chain, residue_number, amino_acid, insertion_code)
            residues[residue_id] = residue
            chain.add_residue(residue)
        else:
            residue = residues[residue_id]

        # Init atom.
        atom = Atom(residue, atom_name, elements_by_symbol[element_symbol], atom_position, occupancy)
        atom = _add_atom_to_residue(atom, residue)
        atoms.add(atom)

    return atoms


def get_residue_distance(residue1, residue2):
    """ Get the shortest distance between two atoms from two different given residues.

        Args:
            residue1(deeprank residue object): the first residue
            residue2(deeprank residue object): the second residue

        Returns(float): the shortest distance
    """

    residue1_atom_positions = numpy.array([atom.position for atom in residue1.atoms])
    residue2_atom_positions = numpy.array([atom.position for atom in residue2.atoms])

    distances = distance_matrix(residue1_atom_positions, residue2_atom_positions, p=2)

    return numpy.min(distances)


def get_residue_contact_pairs(pdb_path, model_id, chain_id1, chain_id2, distance_cutoff):
    """ Get the residues that contact each other at a protein-protein interface.

        Args:
            pdb_path(str): path to the pdb file
            model_id(str): unique identifier for the structure
            chain_id1(str): first protein chain identifier
            chain_id2(str): second protein chain identifier
            distance_cutoff(float): max distance between two interacting residues

        Returns: (list of deeprank residue pairs): the contacting residues
    """

    # load the structure
    pdb = pdb2sql(pdb_path)
    try:
        structure = get_structure(pdb, model_id)
    finally:
        pdb._close()

    # Find out which residues are pairs
    interface = get_interface(pdb_path)
    try:
        contact_residues = interface.get_contact_residues(cutoff=distance_cutoff,
                                                          chain1=chain_id1, chain2=chain_id2,
                                                          return_contact_pairs=True)
    finally:
        interface._close()

    # Map to residue objects
    residue_pairs = set([])
    for residue_key1 in contact_residues:
        residue_chain_id1, residue_number1, residue_name1 = residue_key1

        chain1 = structure.get_chain(residue_chain_id1)

        residue1 = None
        for residue in chain1.residues:
            if residue.number == residue_number1 and residue.amino_acid is not None and residue.amino_acid.three_letter_code == residue_name1:
                residue1 = residue
                break
        else:
            raise ValueError("Not found: {} {} {} {}".format(pdb_ac, residue_chain_id1, residue_number1, residue_name1))

        for residue_chain_id2, residue_number2, residue_name2 in contact_residues[residue_key1]:

            chain2 = structure.get_chain(residue_chain_id2)

            residue2 = None
            for residue in chain2.residues:
                if residue.number == residue_number2 and residue.amino_acid is not None and residue.amino_acid.three_letter_code == residue_name2:
                    residue2 = residue
                    break
            else:
                raise ValueError("Not found: {} {} {} {}".format(pdb_ac, residue_chain_id2, residue_number2, residue_name2))

            residue_pairs.add(Pair(residue1, residue2))

    return residue_pairs


def get_surrounding_residues(pdb_path: str, structure_id: str, chain_id: str,
                             residue_number: int, insertion_code: Optional[str],
                             radius: float) -> List[Residue]:

    """ Get the residues that lie within a radius around a residue.

        Args:
            pdb_path: points to PDB file
            structure_id: name for the structure
            chain_id: identifies the residue in the pdb
            residue_number: identifies the residue in the pdb
            insertion_code: identifies the residue in the pdb
            radius: max distance in Ångström between atoms of the residue and the other residues

        Returns: (a set of deeprank residues): the surrounding residues
    """

    pdb = pdb2sql(pdb_path)

    structure_atom_rows = pdb.get("x,y,z,rowID,name,altLoc,occ,element,chainID,resSeq,resName,iCode", model=0)
    structure_atom_positions = [row[:3] for row in structure_atom_rows]

    # convert insertion code to pdb2sql's format
    if insertion_code is None:
        insertion_code = ""

    residue_atom_positions = pdb.get("x,y,z",
                                     chainID=chain_id,
                                     resSeq=residue_number, iCode=insertion_code,
                                     model=0)

    if len(residue_atom_positions) == 0:
        raise ValueError("residue not found: {} {}{}".format(chain_id, residue_number, insertion_code))

    distances = distance_matrix(structure_atom_positions, residue_atom_positions, p=2)
    neighbours = distances < radius

    neighbour_indices = numpy.nonzero(neighbours)[0]

    neighbour_atom_rows = [structure_atom_rows[i] for i in neighbour_indices]

    atoms = _to_atoms(neighbour_atom_rows, structure_id)

    residues = set([atom.residue for atom in atoms])

    return residues


