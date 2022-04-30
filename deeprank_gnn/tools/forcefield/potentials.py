from typing import List

import numpy

from deeprank_gnn.models.structure import Atom
from deeprank_gnn.models.forcefield.vanderwaals import VanderwaalsParam
from deeprank_gnn.domain.forcefield import (VANDERWAALS_DISTANCE_ON, VANDERWAALS_DISTANCE_OFF,
                                            SQUARED_VANDERWAALS_DISTANCE_ON, SQUARED_VANDERWAALS_DISTANCE_OFF,
                                            EPSILON0, COULOMB_CONSTANT)


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
