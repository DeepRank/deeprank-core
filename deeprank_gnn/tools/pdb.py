from time import time
import logging
from typing import List, Optional
import os

from scipy.spatial import distance_matrix
import numpy
import torch
from pdb2sql import pdb2sql, interface as get_interface

from deeprank_gnn.models.structure import Atom, Residue, Chain, Structure, AtomicElement
from deeprank_gnn.domain.amino_acid import amino_acids
from deeprank_gnn.models.pair import Pair
from deeprank_gnn.domain.forcefield import atomic_forcefield, COULOMB_CONSTANT, EPSILON0
from deeprank_gnn.models.contact import ResidueContact, AtomicContact
from deeprank_gnn.models.forcefield.vanderwaals import VanderwaalsParam


_log = logging.getLogger(__name__)


def is_xray(pdb_file):
    "check that an open pdb file is an x-ray structure"

    for line in pdb_file:
        if line.startswith("EXPDTA") and "X-RAY DIFFRACTION" in line:
            return True

    return False



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


def _to_atoms_in_structure(atom_rows: List, structure_id: str) -> List[Atom]:

    amino_acids_by_code = {amino_acid.three_letter_code: amino_acid for amino_acid in amino_acids}
    elements_by_name = {element.name: element for element in AtomicElement}

    # We need these intermediary dicts to keep track of which residues and chains have already been created.
    chains = {}
    residues = {}

    atoms = set([])
    structure = Structure(id_)

    # Iterate over the atom output from pdb2sql
    for row in pdb.get("x,y,z,rowID,name,altLoc,occ,element,chainID,resSeq,resName,iCode", model=0):

        x, y, z, atom_number, atom_name, altloc, occupancy, element, chain_id, residue_number, residue_name, insertion_code = row

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
        atom = Atom(residue, atom_name, elements_by_name[element], atom_position, occupancy)
        atom = _add_atom_to_residue(atom, residue)
        atoms.add(atom)

    return list(atoms)


def get_coulomb_potentials(distances: numpy.ndarray, charges: List[float]) -> numpy.ndarray:
    """ Calculate the Coulomb potentials, given a distance matrix and a list of charges of equal size.

        Warning: there's no distance cutoff here. The radius of influence is assumed to infinite
    """

    # check for the correct matrix shape
    charge_count = len(charges)
    if charge_count != distances.shape[0] or charge_count != distances.shape[1]:
        raise ValueError("Cannot calculate distances between {} charges and {} distances"
                         .format(charge_count, "x".join(distances.shape)))

    # calculate the potentials
    potentials = numpy.expand_dims(charges, axis=0) * numpy.expand_dims(charges, axis=1) \
                 * COULOMB_CONSTANT / (EPSILON0 * distances)

    return potentials


def get_lennard_jones_potentials(distances: numpy.ndarray, atoms: List[Atom],
                                 vanderwaals_parameters: List[VanderwaalsParam]) -> numpy.ndarray:
    """ Calculate Lennard-Jones potentials, given a distance matrix and a list of atoms with vanderwaals parameters of equal size.

         Warning: there's no distance cutoff here. The radius of influence is assumed to infinite
    """

    # check for the correct data shapes
    atom_count = len(atoms)
    if atom_count != len(vanderwaals_parameters):
        raise ValueError("The number of atoms ({}) does not match the number of vanderwaals parameters ({})"
                         .format(atom_count, len(vanderwaals_parameters)))
    if atom_count != distances.shape[0] or atom_count != distances.shape[1]:
        raise ValueError("Cannot calculate distances between {} atoms and {} distances"
                         .format(atom_count, "x".join(distances.shape)))

    # collect parameters
    sigmas1 = numpy.empty((atom_count, atom_count))
    sigmas2 = numpy.empty((atom_count, atom_count))
    epsilons1 = numpy.empty((atom_count, atom_count))
    epsilons2 = numpy.empty((atom_count, atom_count))
    for atom1_index in range(atom_count):
        for atom2_index in range(atom_count):
            atom1 = atoms[atom1_index]
            atom2 = atoms[atom2_index]

            # Which parameter we take, depends on whether the contact is intra- or intermolecular.
            if atom1.residue != atom2.residue:

                sigmas1[atom1_index][atom2_index] = vanderwaals_parameters[atom1_index].inter_sigma
                sigmas2[atom1_index][atom2_index] = vanderwaals_parameters[atom2_index].inter_sigma

                epsilons1[atom1_index][atom2_index] = vanderwaals_parameters[atom1_index].inter_epsilon
                epsilons2[atom1_index][atom2_index] = vanderwaals_parameters[atom2_index].inter_epsilon
            else:
                sigmas1[atom1_index][atom2_index] = vanderwaals_parameters[atom1_index].intra_sigma
                sigmas2[atom1_index][atom2_index] = vanderwaals_parameters[atom2_index].intra_sigma

                epsilons1[atom1_index][atom2_index] = vanderwaals_parameters[atom1_index].intra_epsilon
                epsilons2[atom1_index][atom2_index] = vanderwaals_parameters[atom2_index].intra_epsilon

    # calculate potentials
    sigmas = 0.5 * (sigmas1 + sigmas2)
    epsilons = numpy.sqrt(sigmas1 * sigmas2)
    potentials = 4.0 * epsilons * ((sigmas / distances) ** 12 - (sigmas / distances) ** 6)

    return potentials


def get_atomic_contacts(atoms: List[Atom]) -> List[AtomicContact]:
    """ Computes all the contacts between a given list of atoms.
        It doesn't pair atoms with themselves, so no zero divisions are expected.

        Warning: do not call this on all atoms in the structure, since this can lead to
                 excessive memory consumption! Be sure to use queries to make a selection.
    """

    atom_postions = [atom.position for atom in atoms]

    # calculate distance matrix
    interatomic_distances = distance_matrix(atom_postions, atom_postions, p=2)

    # get forcefield parameters
    atom_charges = [atomic_forcefield.get_charge(atom) for atom in atoms]
    atom_vanderwaals_parameters = [atomic_forcefield.get_vanderwaals_parameters(atom) for atom in atoms]

    # calculate potentials
    interatomic_electrostatic_potentials = get_coulomb_potentials(interatomic_distances, atom_charges)
    interatomic_vanderwaals_potentials = get_lennard_jones_potentials(interatomic_distances, atoms, atom_vanderwaals_parameters)

    # build contacts
    contacts = []

    count_atoms = len(atoms)
    for atom1_index in range(count_atoms):
        for atom2_index in range(atom1_index + 1, count_atoms):  # don't make the same contact twice and don't pair an atom with itself

            contacts.append(AtomicContact(atoms[atom1_index], atoms[atom2_index],
                                          interatomic_distances[atom1_index][atom2_index],
                                          interatomic_electrostatic_potentials[atom1_index][atom2_index],
                                          interatomic_vanderwaals_potentials[atom1_index][atom2_index]))
    return contacts


def get_residue_contacts(residues: List[Residue]) -> List[ResidueContact]:
    """ Computes all the contacts between a given list of residues.
        It doesn't pair residues with themselves, so no zero divisions are expected.

        Warning: do not call this on all residues in the structure, since this can lead to
                 excessive memory consumption! Be sure to use queries to make a selection.
    """

    atoms = []
    for residue in residues:
        atoms += residue.atoms

    atomic_contacts = get_atomic_contacts(atoms)

    # start with empty dictionaries, to hold the distance and energies per contact
    residue_minimum_distances = {}
    residue_electrostatic_potential_sums = {}
    residue_vanderwaals_potential_sums = {}

    # iterate over all interatomic contacts
    for atomic_contact in atomic_contacts:
        if atomic_contact.atom1.residue != atomic_contact.atom2.residue:  # don't pair a residue with itself

            residue1 = atomic_contact.atom1.residue
            residue2 = atomic_contact.atom2.residue

            residue_pair = Pair(residue1, residue2)

            if residue_pair not in residue_minimum_distances:

                # initialize distance and energies per residue contact
                residue_minimum_distances[residue_pair] = 1e99
                residue_electrostatic_potential_sums[residue_pair] = 0.0
                residue_vanderwaals_potential_sums[residue_pair] = 0.0

            # aggregate
            residue_minimum_distances[residue_pair] = min(residue_minimum_distances[residue_pair], atomic_contact.distance)
            residue_electrostatic_potential_sums[residue_pair] += atomic_contact.electrostatic_potential
            residue_vanderwaals_potential_sums[residue_pair] += atomic_contact.vanderwaals_potential

    # convert to residue contacts
    residue_contacts = []
    for residue_pair, distance in residue_minimum_distances.items():
        residue_contacts.append(ResidueContact(residue_pair.item1, residue_pair.item2,
                                               residue_minimum_distances[residue_pair],
                                               residue_electrostatic_potential_sums[residue_pair],
                                               residue_vanderwaals_potential_sums[residue_pair]))
    return residue_contacts



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



def get_surrounding_residues(pdb_path: str,
                             residue_chain_id: str, residue_number: int, residue_insertion_code: Optional[str],
                             radius: float) -> List[Residue]:
    """ Get the residues that lie within a radius around a residue.
    """

    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]

    pdb = pdb2sql(pdb_path)

    if residue_insertion_code is None:
        residue_insertion_code = ""

    atom_rows_of_residue = pdb.get("x,y,z,rowID,name,altLoc,occ,element,chainID,resSeq,resName,iCode",
                                   model=0, chainID=residue_chain_id, resSeq=residue_number, iCode=residue_insertion_code)
    atom_positions_of_residue = [atom_row[:3] for atom_row in atom_rows_of_residue]

    atom_rows = pdb.get("x,y,z,rowID,name,altLoc,occ,element,chainID,resSeq,resName,iCode", model=0)
    atom_positions = [atom_row[:3] for atom_row in atom_rows]
    distances = distance_matrix(atom_positions_of_residue, atom_positions)
    neighbours = distances < radius

    neighbour_atom_rows = []
    for atom_index in numpy.nonzero(neighbours)[1]:
        atom_row = atom_rows[atom_index]
        neighbour_atom_rows.append(atom_row)

    structure_id = f"{pdb_name}-{radius}-around-{residue_chain_id}{residue_number}{residue_insertion_code}"
    atoms = _to_atoms_in_structure(neighbour_atom_rows, structure_id)

    residues = set([])
    for atom in atoms:
        residues.add(atom.residue)

    return list(residues)


def find_neighbour_atoms(atoms, max_distance):
    """ For a given list of atoms, find the pairs of atoms that lie next to each other.

        Args:
            atoms(list of deeprank atom objects): the atoms to look at
            max_distance(float): max distance between two atoms in Ångström

        Returns: (a set of deeprank atom object pairs): the paired atoms
    """

    atom_count = len(atoms)

    atom_positions = [atom.position for atom in atoms]

    distances = distance_matrix(atom_positions, atom_positions, p=2)

    neighbours = distances < max_distance

    pairs = set([])

    for atom1_index, atom2_index in numpy.transpose(numpy.nonzero(neighbours)):
        if atom1_index != atom2_index:
            pairs.add(Pair(atoms[atom1_index], atoms[atom2_index]))

    return pairs
