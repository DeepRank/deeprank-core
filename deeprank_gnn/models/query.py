import os
from enum import Enum
import logging

import freesasa
import pdb2sql
import numpy
from scipy.spatial import distance_matrix

from deeprank_gnn.models.error import UnknownAtomError
from deeprank_gnn.tools.pssm import parse_pssm
from deeprank_gnn.tools import BioWrappers, BSA
from deeprank_gnn.tools.pdb import get_residue_contact_pairs, get_residue_distance, get_surrounding_residues, get_structure
from deeprank_gnn.models.graph import Graph
from deeprank_gnn.domain.graph import EDGETYPE_INTERNAL, EDGETYPE_INTERFACE
from deeprank_gnn.domain.feature import *
from deeprank_gnn.domain.amino_acid import *
from deeprank_gnn.domain.forcefield import (atomic_forcefield,
                                            VANDERWAALS_DISTANCE_ON, VANDERWAALS_DISTANCE_OFF,
                                            SQUARED_VANDERWAALS_DISTANCE_ON, SQUARED_VANDERWAALS_DISTANCE_OFF,
                                            EPSILON0, COULOMB_CONSTANT)


_log = logging.getLogger(__name__)


class Query:
    """ Represents one entity of interest, like a single residue variant or a protein-protein interface.

        Query objects are used to generate graphs from structures.
        objects of this class should be created before any model is loaded
    """

    def __init__(self, model_id, targets=None):
        """
            Args:
                model_id(str): the id of the model to load, usually a pdb accession code
                targets(dict, optional): target values associated with this query
        """

        self._model_id = model_id

        if targets is None:
            self._targets = {}
        else:
            self._targets = targets

    @property
    def model_id(self):
        return self._model_id

    @property
    def targets(self):
        return self._targets

    def __repr__(self):
        return "{}({})".format(type(self), self.get_query_id())


class SingleResidueVariantResidueQuery(Query):
    "creates a residue graph from a single residue variant in a pdb file"

    def __init__(self, pdb_path, chain_id, residue_number, insertion_code, wildtype_amino_acid, variant_amino_acid,
                 pssm_paths=None,
                 radius=10.0, external_distance_cutoff=4.5, targets=None):

        """
            Args:
                pdb_path(str): the path to the pdb file
                chain_id(str): the pdb chain identifier of the variant residue
                residue_number(int): the number of the variant residue
                insertion_code(str): the insertion code of the variant residue, set to None if not applicable
                wildtype_amino_acid(deeprank amino acid object): the wildtype amino acid
                variant_amino_acid(deeprank amino acid object): the variant amino acid
                pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
                radius(float): in Ångström, determines how many residues will be included in the graph
                external_distance_cutoff(float): max distance in Ångström between a pair of atoms to consider them as an external edge in the graph
                targets(dict(str,float)): named target values associated with this query
        """

        self._pdb_path = pdb_path
        self._pssm_paths = pssm_paths

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wildtype_amino_acid = wildtype_amino_acid
        self._variant_amino_acid = variant_amino_acid

        self._radius = radius
        self._external_distance_cutoff = external_distance_cutoff

    @property
    def residue_id(self):
        "residue identifier within chain"

        if self._insertion_code is not None:

            return "{}{}".format(self._residue_number, self._insertion_code)
        else:
            return str(self._residue_number)

    def get_query_id(self):
        return "residue-graph-{}:{}:{}:{}->{}".format(self.model_id, self._chain_id, self.residue_id, self._wildtype_amino_acid.name, self._variant_amino_acid.name)

    @staticmethod
    def _get_residue_node_key(residue):
        "produce an unique node key, given the residue"

        return str(residue)

    @staticmethod
    def _is_next_residue_number(residue1, residue2):
        if residue1.number == residue2.number:
            if residue1.insertion_code is not None and residue2.insertion_code is not None:
                return (ord(residue1.insertion_code) + 1) == ord(residue2.insertion_code)

        elif (residue1.number + 1) == residue2.number:
            return True

        return False

    @staticmethod
    def _is_covalent_bond(atom1, atom2, distance):
        if distance < 2.3:

            # peptide bonds
            if atom1.name == "C" and atom2.name == "N" and \
                    SingleResidueVariantResidueQuery._is_next_residue_number(atom1.residue, atom2.residue):
                return True

            elif atom2.name == "C" and atom1.name == "N" and \
                    SingleResidueVariantResidueQuery._is_next_residue_number(atom2.residue, atom1.residue):
                return True

            # disulfid bonds
            elif atom1.name == "SG" and atom2.name == "SG":
                return True

        return False

    @staticmethod
    def _set_sasa(graph, node_name_residues, pdb_path):

        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)

        for node_name, residue in node_name_residues.items():

            select_str = ('residue, (resi %s) and (chain %s)' % (residue.number_string, residue.chain.id),)

            area = freesasa.selectArea(select_str, structure, result)['residue']

            if numpy.isnan(area):
                raise ValueError("freesasa returned {} for {}:{}".format(area, pdb_path, residue))

            graph.nodes[node_name][FEATURENAME_SASA] = area

    @staticmethod
    def _set_amino_acid_properties(graph, node_name_residues, variant_residue, wildtype_amino_acid, variant_amino_acid):
        for node_name, residue in node_name_residues.items():
            graph.nodes[node_name][FEATURENAME_POSITION] = numpy.mean([atom.position for atom in residue.atoms], axis=0)
            graph.nodes[node_name][FEATURENAME_CHARGE] = residue.amino_acid.charge
            graph.nodes[node_name][FEATURENAME_POLARITY] = residue.amino_acid.polarity.onehot
            graph.nodes[node_name][FEATURENAME_SIZE] = residue.amino_acid.size

            if residue == variant_residue:

                graph.nodes[node_name][FEATURENAME_AMINOACID] = wildtype_amino_acid.onehot
                graph.nodes[node_name][FEATURENAME_VARIANTAMINOACID] = variant_amino_acid.onehot
                graph.nodes[node_name][FEATURENAME_SIZEDIFFERENCE] = variant_amino_acid.size - wildtype_amino_acid.size
                graph.nodes[node_name][FEATURENAME_POLARITYDIFFERENCE] = variant_amino_acid.polarity.onehot - wildtype_amino_acid.polarity.onehot
            else:
                graph.nodes[node_name][FEATURENAME_AMINOACID] = residue.amino_acid.onehot
                graph.nodes[node_name][FEATURENAME_VARIANTAMINOACID] = numpy.zeros(len(residue.amino_acid.onehot))
                graph.nodes[node_name][FEATURENAME_SIZEDIFFERENCE] = 0
                graph.nodes[node_name][FEATURENAME_POLARITYDIFFERENCE] = numpy.zeros(len(residue.amino_acid.polarity.onehot))

    amino_acid_order = [alanine, arginine, asparagine, aspartate, cysteine, glutamine, glutamate, glycine, histidine, isoleucine,
                        leucine, lysine, methionine, phenylalanine, proline, serine, threonine, tryptophan, tyrosine, valine]

    @staticmethod
    def _set_pssm(graph, node_name_residues, variant_residue, wildtype_amino_acid, variant_amino_acid):

        for node_name, residue in node_name_residues.items():
            pssm_row = residue.get_pssm()

            pssm_value = [pssm_row.conservations[amino_acid] for amino_acid in SingleResidueVariantResidueQuery.amino_acid_order]

            if residue == variant_residue:

                graph.nodes[node_name][FEATURENAME_PSSMDIFFERENCE] = pssm_row.get_conservation(variant_amino_acid) - \
                                                                     pssm_row.get_conservation(wildtype_amino_acid)

                graph.nodes[node_name][FEATURENAME_PSSMWILDTYPE] = pssm_row.get_conservation(wildtype_amino_acid)
                graph.nodes[node_name][FEATURENAME_PSSMVARIANT] = pssm_row.get_conservation(variant_amino_acid)
            else:
                graph.nodes[node_name][FEATURENAME_PSSMDIFFERENCE] = 0.0
                graph.nodes[node_name][FEATURENAME_PSSMWILDTYPE] = pssm_row.get_conservation(residue.amino_acid)
                graph.nodes[node_name][FEATURENAME_PSSMVARIANT] = pssm_row.get_conservation(residue.amino_acid)

            graph.nodes[node_name][FEATURENAME_INFORMATIONCONTENT] = pssm_row.information_content
            graph.nodes[node_name][FEATURENAME_PSSM] = pssm_value

    def build_graph(self):
        # load pdb strucure
        pdb = pdb2sql.pdb2sql(self._pdb_path)

        try:
            structure = get_structure(pdb, self.model_id)
        finally:
            pdb._close()

        # read the pssm
        if self._pssm_paths is not None:
            for chain in structure.chains:
                if chain.id in self._pssm_paths:
                    pssm_path = self._pssm_paths[chain.id]

                    with open(pssm_path, 'rt') as f:
                        chain.pssm = parse_pssm(f, chain)

        # find the variant residue
        variant_residues = [r for r in structure.get_chain(self._chain_id).residues
                            if r.number == self._residue_number and r.insertion_code == self._insertion_code]
        if len(variant_residues) == 0:
            raise ValueError("Residue {}:{} not found in {}".format(self._chain_id, self.residue_id, self._pdb_path))
        variant_residue = variant_residues[0]

        # get the residues and atoms involved
        residues = get_surrounding_residues(structure, variant_residue, self._radius)
        residues.add(variant_residue)
        atoms = []
        for residue in residues:
            if residue.amino_acid is not None:
                atoms.extend(residue.atoms)

        # build a graph and keep track of how we named the nodes
        node_name_residues = {}
        graph = Graph(self.get_query_id(), self.targets)

        # find neighbouring atoms
        atom_positions = [atom.position for atom in atoms]
        distances = distance_matrix(atom_positions, atom_positions, p=2)
        neighbours = numpy.logical_and(distances < self._external_distance_cutoff,
                                       distances > 0.0)

        # iterate over every pair of neighbouring atoms
        for atom1_index, atom2_index in numpy.transpose(numpy.nonzero(neighbours)):
            if atom1_index != atom2_index:  # do not pair an atom with itself

                atom_distance = distances[atom1_index, atom2_index]

                atom1 = atoms[atom1_index]
                atom2 = atoms[atom2_index]

                if atom1.residue != atom2.residue:  # do not connect a residue to itself

                    residue1 = atom1.residue
                    residue2 = atom2.residue

                    residue1_key = self._get_residue_node_key(residue1)
                    residue2_key = self._get_residue_node_key(residue2)

                    node_name_residues[residue1_key] = residue1
                    node_name_residues[residue2_key] = residue2

                    # add the edge if not already
                    if not graph.has_edge(residue1_key, residue2_key):
                        graph.add_edge(residue1_key, residue2_key)
                        graph.edges[residue1_key, residue2_key][FEATURENAME_EDGETYPE] = EDGETYPE_INTERFACE

                    # covalent bond overrided non-covalent
                    if self._is_covalent_bond(atom1, atom2, atom_distance):
                        graph.edges[residue1_key, residue2_key][FEATURENAME_EDGETYPE] = EDGETYPE_INTERNAL

                    # feature to hold whether the two residues are of the same chain
                    graph.edges[residue1_key, residue2_key][FEATURENAME_EDGESAMECHAIN] = float(atom1.residue.chain == atom2.residue.chain)

                    # Make sure we take the shortest distance
                    if FEATURENAME_EDGEDISTANCE not in graph.edges[residue1_key, residue2_key]:
                        graph.edges[residue1_key, residue2_key][FEATURENAME_EDGEDISTANCE] = atom_distance

                    elif graph.edges[residue1_key, residue2_key][FEATURENAME_EDGEDISTANCE] > atom_distance:
                        graph.edges[residue1_key, residue2_key][FEATURENAME_EDGEDISTANCE] = atom_distance

        # set the node features
        self._set_amino_acid_properties(graph, node_name_residues, variant_residue, self._wildtype_amino_acid, self._variant_amino_acid)
        self._set_sasa(graph, node_name_residues, self._pdb_path)

        self._set_pssm(graph, node_name_residues, variant_residue,
                       self._wildtype_amino_acid, self._variant_amino_acid)

        return graph



class SingleResidueVariantAtomicQuery(Query):
    "creates an atomic graph for a single residue variant in a pdb file"

    def __init__(self, pdb_path, chain_id, residue_number, insertion_code, wildtype_amino_acid, variant_amino_acid,
                 pssm_paths=None,
                 radius=10.0, external_distance_cutoff=4.5, internal_distance_cutoff=3.0,
                 targets=None):
        """
            Args:
                pdb_path(str): the path to the pdb file
                chain_id(str): the pdb chain identifier of the variant residue
                residue_number(int): the number of the variant residue
                insertion_code(str): the insertion code of the variant residue, set to None if not applicable
                wildtype_amino_acid(deeprank amino acid object): the wildtype amino acid
                variant_amino_acid(deeprank amino acid object): the variant amino acid
                pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
                radius(float): in Ångström, determines how many residues will be included in the graph
                external_distance_cutoff(float): max distance in Ångström between a pair of atoms to consider them as an external edge in the graph
                internal_distance_cutoff(float): max distance in Ångström between a pair of atoms to consider them as an internal edge in the graph (must be shorter than external)
                targets(dict(str,float)): named target values associated with this query
        """

        self._pdb_path = pdb_path
        self._pssm_paths = pssm_paths

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._chain_id = chain_id
        self._residue_number = residue_number
        self._insertion_code = insertion_code
        self._wildtype_amino_acid = wildtype_amino_acid
        self._variant_amino_acid = variant_amino_acid

        self._radius = radius

        if external_distance_cutoff < internal_distance_cutoff:
            raise ValueError("this query is not supported with internal distance cutoff shorter than external distance cutoff")

        self._external_distance_cutoff = external_distance_cutoff
        self._internal_distance_cutoff = internal_distance_cutoff

    @property
    def residue_id(self):
        "string representation of the residue number and insertion code"

        if self._insertion_code is not None:
            return "{}{}".format(self._residue_number, self._insertion_code)
        else:
            return str(self._residue_number)

    def get_query_id(self):
        return "{}:{}:{}:{}->{}".format(self.model_id, self._chain_id, self.residue_id, self._wildtype_amino_acid.name, self._variant_amino_acid.name)

    def __eq__(self, other):
        return type(self) == type(other) and self.model_id == other.model_id and \
            self._chain_id == other._chain_id and self.residue_id == other.residue_id and \
            self._wildtype_amino_acid == other._wildtype_amino_acid and self._variant_amino_acid == other._variant_amino_acid

    def __hash__(self):
        return hash((self.model_id, self._chain_id, self.residue_id, self._wildtype_amino_acid, self._variant_amino_acid))

    @staticmethod
    def _get_atom_node_key(atom):
        """ Pickle has problems serializing the graph when the nodes are atoms,
            so use this function to generate an unique key for the atom"""

        # This should include the model, chain, residue and atom
        return str(atom)

    def build_graph(self):

        # load pdb strucure
        pdb = pdb2sql.pdb2sql(self._pdb_path)

        try:
            structure = get_structure(pdb, self.model_id)
        finally:
            pdb._close()

        # read the pssm
        if self._pssm_paths is not None:
            for chain in structure.chains:
                if chain.id in self._pssm_paths:
                    pssm_path = self._pssm_paths[chain.id]

                    with open(pssm_path, 'rt') as f:
                        chain.pssm = parse_pssm(f, chain)

        # find the variant residue
        variant_residues = [r for r in structure.get_chain(self._chain_id).residues
                            if r.number == self._residue_number and r.insertion_code == self._insertion_code]
        if len(variant_residues) == 0:
            raise ValueError("Residue {}:{} not found in {}".format(self._chain_id, self.residue_id, self._pdb_path))
        variant_residue = variant_residues[0]

        # get the residues and atoms involved
        residues = get_surrounding_residues(structure, variant_residue, self._radius)
        residues.add(variant_residue)
        atoms = []
        for residue in residues:
            if residue.amino_acid is not None:
                atoms.extend(residue.atoms)

        # build a graph and keep track of how we named the nodes
        node_name_atoms = {}
        graph = Graph(self.get_query_id(), self.targets)

        # find neighbouring atoms
        atom_positions = [atom.position for atom in atoms]
        distances = distance_matrix(atom_positions, atom_positions, p=2)
        neighbours = numpy.logical_and(distances < self._external_distance_cutoff,
                                       distances > 0.0)

        # initialize these recording dictionaries
        atom_vanderwaals_parameters = {}
        atom_charges = {}
        chain_codes = {}

        # give every chain a code
        for chain in structure.chains:
            chain_codes[chain] = len(chain_codes)

        # iterate over every pair of neighbouring atoms
        for atom1_index, atom2_index in numpy.transpose(numpy.nonzero(neighbours)):
            if atom1_index != atom2_index:  # do not pair an atom with itself

                distance = distances[atom1_index, atom2_index]

                atom1 = atoms[atom1_index]
                atom2 = atoms[atom2_index]

                try:
                    for atom in [atom1, atom2]:
                        atom_vanderwaals_parameters[atom] = atomic_forcefield.get_vanderwaals_parameters(atom)
                        atom_charges[atom] = atomic_forcefield.get_charge(atom)

                except UnknownAtomError as e:
                    # if one of the atoms has no forcefield parameters, do not include this edge
                    _log.warning(str(e))
                    continue

                atom1_key = SingleResidueVariantAtomicQuery._get_atom_node_key(atom1)
                atom2_key = SingleResidueVariantAtomicQuery._get_atom_node_key(atom2)

                # connect the atoms and set the distance
                graph.add_edge(atom1_key, atom2_key)

                if distance < self._internal_distance_cutoff:
                    graph.edges[atom1_key, atom2_key][FEATURENAME_EDGETYPE] = EDGETYPE_INTERNAL
                else:
                    graph.edges[atom1_key, atom2_key][FEATURENAME_EDGETYPE] = EDGETYPE_INTERFACE

                graph.edges[atom1_key, atom2_key][FEATURENAME_EDGEDISTANCE] = distance

                graph.edges[atom1_key, atom2_key][FEATURENAME_EDGESAMECHAIN] = float(atom1.residue.chain == atom2.residue.chain)

                # set the positions of the atoms
                graph.nodes[atom1_key][FEATURENAME_POSITION] = atom1.position
                graph.nodes[atom2_key][FEATURENAME_POSITION] = atom2.position
                graph.nodes[atom1_key][FEATURENAME_CHAIN] = chain_codes[atom1.residue.chain]
                graph.nodes[atom2_key][FEATURENAME_CHAIN] = chain_codes[atom2.residue.chain]

                node_name_atoms[atom1_key] = atom1
                node_name_atoms[atom2_key] = atom2

        # set additional features
        SingleResidueVariantAtomicQuery._set_charges(graph, node_name_atoms, atom_charges)
        SingleResidueVariantAtomicQuery._set_coulomb(graph, node_name_atoms, atom_charges, self._external_distance_cutoff)
        SingleResidueVariantAtomicQuery._set_vanderwaals(graph, node_name_atoms, atom_vanderwaals_parameters)

        SingleResidueVariantAtomicQuery._set_pssm(graph, node_name_atoms, variant_residue,
                                                  self._wildtype_amino_acid, self._variant_amino_acid)

        SingleResidueVariantAtomicQuery._set_sasa(graph, node_name_atoms, self._pdb_path)

        SingleResidueVariantAtomicQuery._set_amino_acid(graph, node_name_atoms, variant_residue, self._wildtype_amino_acid, self._variant_amino_acid)

        return graph

    amino_acid_order = [alanine, arginine, asparagine, aspartate, cysteine, glutamine, glutamate, glycine, histidine, isoleucine,
                        leucine, lysine, methionine, phenylalanine, proline, serine, threonine, tryptophan, tyrosine, valine]

    @staticmethod
    def _set_amino_acid(graph, node_name_atoms, variant_residue, wildtype_amino_acid, variant_amino_acid):

        for node_name, atom in node_name_atoms.items():

            if atom.residue == variant_residue:

                graph.nodes[node_name][FEATURENAME_AMINOACID] = wildtype_amino_acid.onehot
                graph.nodes[node_name][FEATURENAME_VARIANTAMINOACID] = variant_amino_acid.onehot
                graph.nodes[node_name][FEATURENAME_SIZEDIFFERENCE] = variant_amino_acid.size - wildtype_amino_acid.size
                graph.nodes[node_name][FEATURENAME_POLARITYDIFFERENCE] = variant_amino_acid.polarity.onehot - wildtype_amino_acid.polarity.onehot
            else:
                graph.nodes[node_name][FEATURENAME_AMINOACID] = atom.residue.amino_acid.onehot
                graph.nodes[node_name][FEATURENAME_VARIANTAMINOACID] = numpy.zeros(len(atom.residue.amino_acid.onehot))
                graph.nodes[node_name][FEATURENAME_SIZEDIFFERENCE] = 0
                graph.nodes[node_name][FEATURENAME_POLARITYDIFFERENCE] = numpy.zeros(len(atom.residue.amino_acid.polarity.onehot))


    @staticmethod
    def _set_pssm(graph, node_name_atoms, variant_residue, wildtype_amino_acid, variant_amino_acid):

        for node_name, atom in node_name_atoms.items():
            pssm_row = atom.residue.get_pssm()

            pssm_value = [pssm_row.conservations[amino_acid] for amino_acid in SingleResidueVariantAtomicQuery.amino_acid_order]

            if atom.residue == variant_residue:

                graph.nodes[node_name][FEATURENAME_PSSMDIFFERENCE] = pssm_row.get_conservation(variant_amino_acid) - \
                                                                     pssm_row.get_conservation(wildtype_amino_acid)

                graph.nodes[node_name][FEATURENAME_PSSMWILDTYPE] = pssm_row.get_conservation(wildtype_amino_acid)
                graph.nodes[node_name][FEATURENAME_PSSMVARIANT] = pssm_row.get_conservation(variant_amino_acid)
            else:
                graph.nodes[node_name][FEATURENAME_PSSMDIFFERENCE] = 0.0
                graph.nodes[node_name][FEATURENAME_PSSMWILDTYPE] = pssm_row.get_conservation(atom.residue.amino_acid)
                graph.nodes[node_name][FEATURENAME_PSSMVARIANT] = pssm_row.get_conservation(atom.residue.amino_acid)

            graph.nodes[node_name][FEATURENAME_INFORMATIONCONTENT] = pssm_row.information_content
            graph.nodes[node_name][FEATURENAME_PSSM] = pssm_value

    @staticmethod
    def _set_sasa(graph, node_name_atoms, pdb_path):

        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)

        for node_name, atom in node_name_atoms.items():

            if atom.element == "H":  # freeSASA doesn't have these
                area = 0.0
            else:
                select_str = ('atom, (name %s) and (resi %s) and (chain %s)' % (atom.name, atom.residue.number_string, atom.residue.chain.id),)
                area = freesasa.selectArea(select_str, structure, result)['atom']

            if numpy.isnan(area):
                raise ValueError("freesasa returned {} for {}:{}".format(area, pdb_path, atom))

            graph.nodes[node_name][FEATURENAME_SASA] = area

    @staticmethod
    def _set_charges(graph, node_name_atoms, charges):
        for node_name in graph.nodes:
            atom = node_name_atoms[node_name]
            graph.nodes[node_name][FEATURENAME_CHARGE] = charges[atom]


    @staticmethod
    def _set_coulomb(graph, node_name_atoms, charges, max_interatomic_distance):

        # get the edges
        edge_keys = []
        edge_count = len(graph.edges)
        charges1 = numpy.empty(edge_count)
        charges2 = numpy.empty(edge_count)
        edge_distances = numpy.empty(edge_count)
        for edge_index, ((atom1_name, atom2_name), edge) in enumerate(graph.edges.items()):
            atom1 = node_name_atoms[atom1_name]
            atom2 = node_name_atoms[atom2_name]
            edge_keys.append((atom1_name, atom2_name))

            charges1[edge_index] = charges[atom1]
            charges2[edge_index] = charges[atom2]

            edge_distances[edge_index] = edge[FEATURENAME_EDGEDISTANCE]

        # calculate coulomb potentials
        coulomb_constant_factor = COULOMB_CONSTANT / EPSILON0

        coulomb_radius_factors = numpy.square(numpy.ones(edge_count) - numpy.square(edge_distances / max_interatomic_distance))

        coulomb_potentials = charges1 * charges2 * coulomb_constant_factor / edge_distances * coulomb_radius_factors

        # set the values to the edges
        for index, potential in enumerate(coulomb_potentials):
            graph.edges[edge_keys[index]][FEATURENAME_EDGECOULOMB] = potential

    @staticmethod
    def _set_vanderwaals(graph, node_name_atoms, vanderwaals_parameters):

        edge_keys = []
        edge_count = len(graph.edges)
        sigmas = numpy.empty(edge_count)
        epsilons = numpy.empty(edge_count)
        edge_distances = numpy.empty(edge_count)
        for edge_index, ((atom1_name, atom2_name), edge) in enumerate(graph.edges.items()):
            atom1 = node_name_atoms[atom1_name]
            atom2 = node_name_atoms[atom2_name]
            edge_keys.append((atom1_name, atom2_name))

            vanderwaals_parameters1 = vanderwaals_parameters[atom1]
            vanderwaals_parameters2 = vanderwaals_parameters[atom2]

            if atom1.residue.chain != atom2.residue.chain:

                # intermolecular
                sigma1 = vanderwaals_parameters1.inter_sigma
                sigma2 = vanderwaals_parameters2.inter_sigma
                epsilon1 = vanderwaals_parameters1.inter_epsilon
                epsilon2 = vanderwaals_parameters2.inter_epsilon
            else:
                # intramolecular
                sigma1 = vanderwaals_parameters1.intra_sigma
                sigma2 = vanderwaals_parameters2.intra_sigma
                epsilon1 = vanderwaals_parameters1.intra_epsilon
                epsilon2 = vanderwaals_parameters2.intra_epsilon

            sigmas[edge_index] = 0.5 * (sigma1 + sigma2)
            epsilons[edge_index] = numpy.sqrt(epsilon1 * epsilon2)

            edge_distances[edge_index] = edge[FEATURENAME_EDGEDISTANCE]

        # calculate potentials
        vanderwaals_constant_factor = (SQUARED_VANDERWAALS_DISTANCE_OFF - SQUARED_VANDERWAALS_DISTANCE_ON) ** 3

        indices_tooclose = numpy.nonzero(edge_distances < VANDERWAALS_DISTANCE_ON)
        indices_toofar = numpy.nonzero(edge_distances > VANDERWAALS_DISTANCE_OFF)

        squared_distances = numpy.square(edge_distances)

        vanderwaals_prefactors = (((SQUARED_VANDERWAALS_DISTANCE_OFF - squared_distances) ** 2) *
                                  (SQUARED_VANDERWAALS_DISTANCE_OFF - squared_distances - 3 *
                                  (SQUARED_VANDERWAALS_DISTANCE_ON - squared_distances)) / vanderwaals_constant_factor)
        vanderwaals_prefactors[indices_tooclose] = 0.0
        vanderwaals_prefactors[indices_toofar] = 1.0

        vanderwaals_potentials = 4.0 * epsilons * (((sigmas / edge_distances) ** 12) - ((sigmas / edge_distances) ** 6)) * vanderwaals_prefactors

        # set the values to the edges
        for index, potential in enumerate(vanderwaals_potentials):
            graph.edges[edge_keys[index]][FEATURENAME_EDGEVANDERWAALS] = potential


class ProteinProteinInterfaceResidueQuery(Query):
    "a query that builds residue-based graphs, using the residues at a protein-protein interface"

    def __init__(self, pdb_path, chain_id1, chain_id2, pssm_paths=None,
                 interface_distance_cutoff=8.5, internal_distance_cutoff=3.0,
                 use_biopython=False, targets=None):
        """
            Args:
                pdb_path(str): the path to the pdb file
                chain_id1(str): the pdb chain identifier of the first protein of interest
                chain_id2(str): the pdb chain identifier of the second protein of interest
                pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
                interface_distance_cutoff(float): max distance in Ångström between two interacting residues of the two proteins
                internal_distance_cutoff(float): max distance in Ångström between two interacting residues within the same protein
                use_biopython(bool): whether or not to use biopython tools
                targets(dict, optional): named target values associated with this query
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._interface_distance_cutoff = interface_distance_cutoff
        self._internal_distance_cutoff = internal_distance_cutoff

        self._use_biopython = use_biopython

    def get_query_id(self):
        return "{}:{}-{}".format(self.model_id, self._chain_id1, self._chain_id2)

    def __eq__(self, other):
        return type(self) == type(other) and self.model_id == other.model_id and \
            {self._chain_id1, self._chain_id2} == {other._chain_id1, other._chain_id2}

    def __hash__(self):
        return hash((self.model_id, tuple(sorted([self._chain_id1, self._chain_id2]))))

    @staticmethod
    def _residue_is_valid(residue):
        if residue.amino_acid is None:
            return False

        if residue not in residue.chain.pssm:
            _log.debug("{} not in pssm".format(residue))
            return False

        return True

    @staticmethod
    def _get_residue_node_key(residue):
        """ Pickle has trouble serializing a graph if the keys are residue objects, so
            we map the residues to string identifiers
        """

        # this should include everything to identify the residue: structure, chain, number, insertion_code
        return str(residue)

    def build_graph(self):
        """ Builds the residue graph.

            Returns(deeprank graph object): the resulting graph
        """

        # get residues from the pdb

        interface_pairs = get_residue_contact_pairs(self._pdb_path, self.model_id,
                                                    self._chain_id1, self._chain_id2,
                                                    self._interface_distance_cutoff)
        if len(interface_pairs) == 0:
            raise ValueError("no interface residues found")

        interface_pairs_list = list(interface_pairs)

        model = interface_pairs_list[0].item1.chain.model
        chain1 = model.get_chain(self._chain_id1)
        chain2 = model.get_chain(self._chain_id2)

        # read the pssm
        if self._pssm_paths is not None:
            for chain in (chain1, chain2):
                pssm_path = self._pssm_paths[chain.id]

                with open(pssm_path, 'rt') as f:
                    chain.pssm = parse_pssm(f, chain)

        # separate residues by chain
        residues_from_chain1 = set([])
        residues_from_chain2 = set([])
        for residue1, residue2 in interface_pairs:

            if self._residue_is_valid(residue1):
                if residue1.chain.id == self._chain_id1:
                    residues_from_chain1.add(residue1)

                elif residue1.chain.id == self._chain_id2:
                    residues_from_chain2.add(residue1)

            if self._residue_is_valid(residue2):
                if residue2.chain.id == self._chain_id1:
                    residues_from_chain1.add(residue2)

                elif residue2.chain.id == self._chain_id2:
                    residues_from_chain2.add(residue2)

        # create the graph
        graph = Graph(self.get_query_id(), self.targets)

        # These will not be stored in the graph, but we need them to get features.
        residues_by_node = {}

        # interface edges
        for pair in interface_pairs:

            residue1, residue2 = pair
            if self._residue_is_valid(residue1) and self._residue_is_valid(residue2):

                distance = get_residue_distance(residue1, residue2)

                key1 = ProteinProteinInterfaceResidueQuery._get_residue_node_key(residue1)
                key2 = ProteinProteinInterfaceResidueQuery._get_residue_node_key(residue2)

                residues_by_node[key1] = residue1
                residues_by_node[key2] = residue2

                graph.add_edge(key1, key2)
                graph.edges[key1, key2][FEATURENAME_EDGEDISTANCE] = distance
                graph.edges[key1, key2][FEATURENAME_EDGETYPE] = EDGETYPE_INTERFACE

        # internal edges
        for residue_set in (residues_from_chain1, residues_from_chain2):
            residue_list = list(residue_set)
            for index, residue1 in enumerate(residue_list):
                for residue2 in residue_list[index + 1:]:
                    distance = get_residue_distance(residue1, residue2)

                    if distance < self._internal_distance_cutoff:
                        key1 = ProteinProteinInterfaceResidueQuery._get_residue_node_key(residue1)
                        key2 = ProteinProteinInterfaceResidueQuery._get_residue_node_key(residue2)

                        residues_by_node[key1] = residue1
                        residues_by_node[key2] = residue2

                        graph.add_edge(key1, key2)
                        graph.edges[key1, key2][FEATURENAME_EDGEDISTANCE] = distance
                        graph.edges[key1, key2][FEATURENAME_EDGETYPE] = EDGETYPE_INTERNAL

        # get bsa
        pdb = pdb2sql.interface(self._pdb_path)
        try:
            bsa_calc = BSA.BSA(self._pdb_path, pdb)
            bsa_calc.get_structure()
            bsa_calc.get_contact_residue_sasa(cutoff=self._interface_distance_cutoff)
            bsa_data = bsa_calc.bsa_data
        finally:
            pdb._close()

        # get biopython features
        if self._use_biopython:
            bio_model = BioWrappers.get_bio_model(self._pdb_path)
            residue_depths = BioWrappers.get_depth_contact_res(bio_model,
                                                               [(residue.chain.id, residue.number, residue.amino_acid.three_letter_code)
                                                                for residue in residues_by_node.values()])
            hse = BioWrappers.get_hse(bio_model)

        # add node features
        chain_codes = {chain1: 0.0, chain2: 1.0}
        for node_key, node in graph.nodes.items():
            residue = residues_by_node[node_key]

            bsa_key = (residue.chain.id, residue.number, residue.amino_acid.three_letter_code)
            bio_key = (residue.chain.id, residue.number)

            pssm_row = residue.get_pssm()
            pssm_value = [pssm_row.conservations[amino_acid] for amino_acid in self.amino_acid_order]

            node[FEATURENAME_CHAIN] = chain_codes[residue.chain]
            node[FEATURENAME_POSITION] = numpy.mean([atom.position for atom in residue.atoms], axis=0)
            node[FEATURENAME_AMINOACID] = residue.amino_acid.onehot
            node[FEATURENAME_CHARGE] = residue.amino_acid.charge
            node[FEATURENAME_POLARITY] = residue.amino_acid.polarity.onehot
            node[FEATURENAME_BURIEDSURFACEAREA] = bsa_data[bsa_key]

            if self._pssm_paths is not None:
                node[FEATURENAME_PSSM] = pssm_value
                node[FEATURENAME_CONSERVATION] = pssm_row.conservations[residue.amino_acid]
                node[FEATURENAME_INFORMATIONCONTENT] = pssm_row.information_content

            if self._use_biopython:
                node[FEATURENAME_RESIDUEDEPTH] = residue_depths[residue] if residue in residue_depths else 0.0
                node[FEATURENAME_HALFSPHEREEXPOSURE] = hse[bio_key] if bio_key in hse else (0.0, 0.0, 0.0)

        return graph

    amino_acid_order = [alanine, arginine, asparagine, aspartate, cysteine, glutamine, glutamate, glycine, histidine, isoleucine,
                        leucine, lysine, methionine, phenylalanine, proline, serine, threonine, tryptophan, tyrosine, valine]


class QueryDataset:
    "represents a collection of data queries"

    def __init__(self):
        self._queries = []

    def add(self, query):
        self._queries.append(query)

    @property
    def queries(self):
        return self._queries

    def __contains__(self, query):
        return query in self._queries

    def __iter__(self):
        return iter(self._queries)
