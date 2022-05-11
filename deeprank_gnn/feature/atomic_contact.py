from typing import List
import logging

import numpy
from scipy.spatial import distance_matrix

from deeprank_gnn.models.structure import Atom
from deeprank_gnn.models.graph import Graph, Edge
from deeprank_gnn.models.contact import ResidueContact, AtomicContact
from deeprank_gnn.domain.feature import (FEATURENAME_EDGEDISTANCE, FEATURENAME_EDGEVANDERWAALS,
                                         FEATURENAME_EDGECOULOMB, FEATURENAME_COVALENT)
from deeprank_gnn.domain.forcefield import atomic_forcefield, COULOMB_CONSTANT, EPSILON0, MAX_COVALENT_DISTANCE
from deeprank_gnn.models.forcefield.vanderwaals import VanderwaalsParam
from deeprank_gnn.models.error import UnknownAtomError


_log = logging.getLogger(__name__)


def get_coulomb_potentials(distances: numpy.ndarray, charges: List[float]) -> numpy.ndarray:
    """ Calculate the Coulomb potentials, given a distance matrix and a list of charges of equal size.

        Warning: there's no distance cutoff here. The radius of influence is assumed to infinite
    """

    # check for the correct matrix shape
    charge_count = len(charges)
    if charge_count != distances.shape[0] or charge_count != distances.shape[1]:
        raise ValueError("Cannot calculate potentials between {} charges and {} distances"
                         .format(charge_count, "x".join([str(d) for d in distances.shape])))

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
        raise ValueError("Cannot calculate potentials between {} atoms and {} distances"
                         .format(atom_count, "x".join([str(d) for d in distances.shape])))

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


def add_features_for_atoms(edges: List[Edge]):

    # get a set of all the atoms involved
    atoms = set([])
    for edge in edges:
        contact = edge.id
        atoms.add(contact.atom1)
        atoms.add(contact.atom2)
    atoms = list(atoms)

    # get all atomic parameters
    atom_indices = {}
    positions = []
    atom_charges = []
    atom_vanderwaals_parameters = []
    for atom_index, atom in enumerate(atoms):
        try:
            charge = atomic_forcefield.get_charge(atom)
            vanderwaals = atomic_forcefield.get_vanderwaals_parameters(atom)

        except UnknownAtomError:
            _log.warning(f"Ignoring atom {atom}, because it's unknown to the forcefield")

            # set parameters to zero, so that the potential becomes zero
            charge = 0.0
            vanderwaals = VanderwaalsParam(0.0, 0.0, 0.0, 0.0)

        atom_charges.append(charge)
        atom_vanderwaals_parameters.append(vanderwaals)
        positions.append(atom.position)
        atom_indices[atom] = atom_index

    # calculate the distance matrix for those atoms
    interatomic_distances = distance_matrix(positions, positions, p=2)

    # calculate potentials
    interatomic_electrostatic_potentials = get_coulomb_potentials(interatomic_distances, atom_charges)
    interatomic_vanderwaals_potentials = get_lennard_jones_potentials(interatomic_distances, atoms, atom_vanderwaals_parameters)

    # determine which atoms are close enough to form a covalent bond
    covalent_neighbours = interatomic_distances < MAX_COVALENT_DISTANCE

    # set the features
    for edge_index, edge in enumerate(edges):
        contact = edge.id
        atom1_index = atom_indices[contact.atom1]
        atom2_index = atom_indices[contact.atom2]

        edge.features[FEATURENAME_EDGEDISTANCE] = interatomic_distances[atom1_index, atom2_index]
        edge.features[FEATURENAME_EDGEVANDERWAALS] = interatomic_vanderwaals_potentials[atom1_index, atom2_index]
        edge.features[FEATURENAME_EDGECOULOMB] = interatomic_electrostatic_potentials[atom1_index, atom2_index]

        if covalent_neighbours[atom1_index, atom2_index]:
            edge.features[FEATURENAME_COVALENT] = 1.0
        else:
            edge.features[FEATURENAME_COVALENT] = 0.0


def add_features_for_residues(edges: List[Edge]):
    # get a set of all the atoms involved
    atoms = set([])
    for edge in edges:
        contact = edge.id
        for atom in (contact.residue1.atoms + contact.residue2.atoms):
            atoms.add(atom)
    atoms = list(atoms)

    # get all atomic parameters
    atom_indices = {}
    positions = []
    atom_charges = []
    atom_vanderwaals_parameters = []
    for atom_index, atom in enumerate(atoms):

        try:
            charge = atomic_forcefield.get_charge(atom)
            vanderwaals = atomic_forcefield.get_vanderwaals_parameters(atom)

        except UnknownAtomError:
            _log.warning(f"Ignoring atom {atom}, because it's unknown to the forcefield")

            # set parameters to zero, so that the potential becomes zero
            charge = 0.0
            vanderwaals = VanderwaalsParam(0.0, 0.0, 0.0, 0.0)

        atom_charges.append(charge)
        atom_vanderwaals_parameters.append(vanderwaals)
        positions.append(atom.position)
        atom_indices[atom] = atom_index

    # calculate the distance matrix for those atoms
    interatomic_distances = distance_matrix(positions, positions, p=2)

    # calculate potentials
    interatomic_electrostatic_potentials = get_coulomb_potentials(interatomic_distances, atom_charges)
    interatomic_vanderwaals_potentials = get_lennard_jones_potentials(interatomic_distances, atoms, atom_vanderwaals_parameters)

    # determine which atoms are close enough to form a covalent bond
    covalent_neighbours = interatomic_distances < MAX_COVALENT_DISTANCE

    # set the features
    for edge_index, edge in enumerate(edges):
        contact = edge.id
        for atom1 in contact.residue1.atoms:
            for atom2 in contact.residue2.atoms:

                atom1_index = atom_indices[atom1]
                atom2_index = atom_indices[atom2]

                edge.features[FEATURENAME_EDGEDISTANCE] = min(edge.features.get(FEATURENAME_EDGEDISTANCE, 1e99),
                                                              interatomic_distances[atom1_index, atom2_index])

                edge.features[FEATURENAME_EDGEVANDERWAALS] = (edge.features.get(FEATURENAME_EDGEVANDERWAALS, 0.0) +
                                                              interatomic_vanderwaals_potentials[atom1_index, atom2_index])

                edge.features[FEATURENAME_EDGECOULOMB] = (edge.features.get(FEATURENAME_EDGECOULOMB, 0.0) +
                                                          interatomic_electrostatic_potentials[atom1_index, atom2_index])

                if covalent_neighbours[atom1_index, atom2_index]:
                    edge.features[FEATURENAME_COVALENT] = 1.0

                elif FEATURENAME_COVALENT not in edge.features:
                    edge.features[FEATURENAME_COVALENT] = 0.0


def add_features(pdb_path: str, graph: Graph, *args, **kwargs):

    if type(graph.edges[0].id) == ResidueContact:
        add_features_for_residues(graph.edges)

    elif type(graph.edges[0].id) == AtomicContact:
        add_features_for_atoms(graph.edges)
    else:
        raise TypeError("Unexpected edge type: {}".format(type(edges[0].id)))




