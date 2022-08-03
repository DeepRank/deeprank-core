import logging
from typing import List
import subprocess

from scipy.spatial import distance_matrix
import numpy
from pdb2sql import interface as get_interface

from deeprankcore.models.structure import Atom, Residue, Chain, Structure, AtomicElement
from deeprankcore.domain.amino_acid import amino_acids
from deeprankcore.models.pair import Pair
from deeprankcore.domain.forcefield import atomic_forcefield
from deeprankcore.models.contact import ResidueContact, AtomicContact
from deeprankcore.feature.atomic_contact import get_coulomb_potentials, get_lennard_jones_potentials


_log = logging.getLogger(__name__)


def is_xray(pdb_file):
    "check that an open pdb file is an x-ray structure"

    for line in pdb_file:
        if line.startswith("EXPDTA") and "X-RAY DIFFRACTION" in line:
            return True

    return False


def add_hydrogens(input_pdb_path, output_pdb_path):
    "this requires reduce: https://github.com/rlabduke/reduce"

    with open(output_pdb_path, "wt", encoding = "utf-8") as f:
        p = subprocess.run(["reduce", input_pdb_path], stdout=subprocess.PIPE, check=True)
        for line in p.stdout.decode().split("\n"):
            f.write(line.replace("   new", "").replace("   std", "") + "\n")


def _add_atom_to_residue(atom, residue):

    for other_atom in residue.atoms:
        if other_atom.name == atom.name:
            # Don't allow two atoms with the same name, pick the highest
            # occupancy
            if other_atom.occupancy < atom.occupancy:
                other_atom.change_altloc(atom)
                return

    # not there yet, add it
    residue.add_atom(atom)


def get_structure(pdb, id_): # pylint: disable=too-many-locals
    """Builds a structure from rows in a pdb file
    Args:
        pdb (pdb2sql object): the pdb structure that we're investigating
        id (str): unique id for the pdb structure
    Returns (Structure): the structure object, giving access to chains, residues, atoms
    """

    amino_acids_by_code = {
        amino_acid.three_letter_code: amino_acid for amino_acid in amino_acids
    }
    elements_by_name = {element.name: element for element in AtomicElement}

    # We need these intermediary dicts to keep track of which residues and
    # chains have already been created.
    chains = {}
    residues = {}

    structure = Structure(id_)

    # Iterate over the atom output from pdb2sql
    for row in pdb.get(
        "x,y,z,rowID,name,altLoc,occ,element,chainID,resSeq,resName,iCode", model=0
    ):

        (
            x,
            y,
            z,
            _,
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
        atom = Atom(
            residue, atom_name, elements_by_name[element], atom_position, occupancy
        )
        _add_atom_to_residue(atom, residue)

    return structure


def get_atomic_contacts(atoms: List[Atom]) -> List[AtomicContact]:
    """Computes all the contacts between a given list of atoms.
    It doesn't pair atoms with themselves, so no zero divisions are expected.

    Warning: do not call this on all atoms in the structure, since this can lead to
             excessive memory consumption! Be sure to use queries to make a selection.
    """

    atom_postions = [atom.position for atom in atoms]

    # calculate distance matrix
    interatomic_distances = distance_matrix(atom_postions, atom_postions, p=2)

    # get forcefield parameters
    atom_charges = [atomic_forcefield.get_charge(atom) for atom in atoms]
    atom_vanderwaals_parameters = [
        atomic_forcefield.get_vanderwaals_parameters(atom) for atom in atoms
    ]

    # calculate potentials
    interatomic_electrostatic_potentials = get_coulomb_potentials(
        interatomic_distances, atom_charges
    )
    interatomic_vanderwaals_potentials = get_lennard_jones_potentials(
        interatomic_distances, atoms, atom_vanderwaals_parameters
    )

    # build contacts
    contacts = []

    count_atoms = len(atoms)
    for atom1_index in range(count_atoms):
        for atom2_index in range(
            atom1_index + 1, count_atoms
        ):  # don't make the same contact twice and don't pair an atom with itself

            contacts.append(
                AtomicContact( # pylint: disable=too-many-function-args
                    atoms[atom1_index],
                    atoms[atom2_index],
                    interatomic_distances[atom1_index][atom2_index],
                    interatomic_electrostatic_potentials[atom1_index][atom2_index],
                    interatomic_vanderwaals_potentials[atom1_index][atom2_index],
                )
            )
    return contacts


def get_residue_contacts(residues: List[Residue]) -> List[ResidueContact]:
    """Computes all the contacts between a given list of residues.
    It doesn't pair residues with themselves, so no zero divisions are expected.

    Warning: do not call this on all residues in the structure, since this can lead to
             excessive memory consumption! Be sure to use queries to make a selection.
    """

    atoms = []
    for residue in residues:
        atoms += residue.atoms

    atomic_contacts = get_atomic_contacts(atoms)

    # start with empty dictionaries, to hold the distance and energies per
    # contact
    residue_minimum_distances = {}
    residue_electrostatic_potential_sums = {}
    residue_vanderwaals_potential_sums = {}

    # iterate over all interatomic contacts
    for atomic_contact in atomic_contacts:
        if (
            atomic_contact.atom1.residue != atomic_contact.atom2.residue
        ):  # don't pair a residue with itself

            residue1 = atomic_contact.atom1.residue
            residue2 = atomic_contact.atom2.residue

            residue_pair = Pair(residue1, residue2)

            if residue_pair not in residue_minimum_distances:

                # initialize distance and energies per residue contact
                residue_minimum_distances[residue_pair] = 1e99
                residue_electrostatic_potential_sums[residue_pair] = 0.0
                residue_vanderwaals_potential_sums[residue_pair] = 0.0

            # aggregate
            residue_minimum_distances[residue_pair] = min(
                residue_minimum_distances[residue_pair], atomic_contact.distance
            )
            residue_electrostatic_potential_sums[
                residue_pair
            ] += atomic_contact.electrostatic_potential
            residue_vanderwaals_potential_sums[
                residue_pair
            ] += atomic_contact.vanderwaals_potential

    # convert to residue contacts
    residue_contacts = []
    for residue_pair, distance in residue_minimum_distances.items():
        residue_contacts.append(
            ResidueContact( # pylint: disable=too-many-function-args
                residue_pair.item1,
                residue_pair.item2,
                distance,
                residue_electrostatic_potential_sums[residue_pair],
                residue_vanderwaals_potential_sums[residue_pair],
            )
        )
    return residue_contacts


def get_residue_distance(residue1, residue2):
    """Get the shortest distance between two atoms from two different given residues.

    Args:
        residue1(deeprank residue object): the first residue
        residue2(deeprank residue object): the second residue

    Returns(float): the shortest distance
    """

    residue1_atom_positions = numpy.array([atom.position for atom in residue1.atoms])
    residue2_atom_positions = numpy.array([atom.position for atom in residue2.atoms])

    distances = distance_matrix(residue1_atom_positions, residue2_atom_positions, p=2)

    return numpy.min(distances)


def get_residue_contact_pairs( # pylint: disable=too-many-locals
    pdb_path: str,
    structure: Structure,
    chain_id1: str,
    chain_id2: str,
    distance_cutoff: float,
) -> List[Pair]:
    """Get the residues that contact each other at a protein-protein interface.

    Args:
        pdb_path: the path of the pdb file, that the structure was built from
        structure: from which to take the residues
        chain_id1: first protein chain identifier
        chain_id2: second protein chain identifier
        distance_cutoff: max distance between two interacting residues

    Returns: the pairs of contacting residues
    """

    # Find out which residues are pairs
    interface = get_interface(pdb_path)
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
    residue_pairs = set([])
    for residue_key1, _ in contact_residues.items():
        residue_chain_id1, residue_number1, residue_name1 = residue_key1

        chain1 = structure.get_chain(residue_chain_id1)

        residue1 = None
        for residue in chain1.residues:
            if (
                residue.number == residue_number1
                and residue.amino_acid is not None
                and residue.amino_acid.three_letter_code == residue_name1
            ):
                residue1 = residue
                break
        else:
            raise ValueError(
                f"Not found: {pdb_path} {residue_chain_id1} {residue_number1} {residue_name1}"
            )

        for residue_chain_id2, residue_number2, residue_name2 in contact_residues[ # pylint: disable=unnecessary-dict-index-lookup
            residue_key1
        ]:

            chain2 = structure.get_chain(residue_chain_id2)

            residue2 = None
            for residue in chain2.residues:
                if (
                    residue.number == residue_number2
                    and residue.amino_acid is not None
                    and residue.amino_acid.three_letter_code == residue_name2
                ):
                    residue2 = residue
                    break
            else:
                raise ValueError(
                    f"Not found: {pdb_path} {residue_chain_id2} {residue_number2} {residue_name2}"
                )

            residue_pairs.add(Pair(residue1, residue2))

    return residue_pairs


def get_surrounding_residues(structure, residue, radius):
    """Get the residues that lie within a radius around a residue.

    Args:
        structure(deeprank structure object): the structure to take residues from
        residue(deeprank residue object): the residue in the structure
        radius(float): max distance in Ångström between atoms of the residue and the other residues

    Returns: (a set of deeprank residues): the surrounding residues
    """

    structure_atoms = structure.get_atoms()
    residue_atoms = residue.atoms

    structure_atom_positions = [atom.position for atom in structure_atoms]
    residue_atom_positions = [atom.position for atom in residue_atoms]

    distances = distance_matrix(structure_atom_positions, residue_atom_positions, p=2)

    close_residues = set([])

    for structure_atom_index, structure_atom in enumerate(structure_atoms):

        shortest_distance = numpy.min(distances[structure_atom_index, :])

        if shortest_distance < radius:

            close_residues.add(structure_atom.residue)

    return close_residues


def find_neighbour_atoms(atoms, max_distance):
    """For a given list of atoms, find the pairs of atoms that lie next to each other.

    Args:
        atoms(list of deeprank atom objects): the atoms to look at
        max_distance(float): max distance between two atoms in Ångström

    Returns: (a set of deeprank atom object pairs): the paired atoms
    """

    atom_positions = [atom.position for atom in atoms]

    distances = distance_matrix(atom_positions, atom_positions, p=2)

    neighbours = distances < max_distance

    pairs = set([])

    for atom1_index, atom2_index in numpy.transpose(numpy.nonzero(neighbours)):
        if atom1_index != atom2_index:
            pairs.add(Pair(atoms[atom1_index], atoms[atom2_index]))

    return pairs