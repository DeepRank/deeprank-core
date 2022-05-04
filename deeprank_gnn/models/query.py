import logging
<<<<<<< HEAD
from tempfile import mkstemp
=======
import os
from typing import Dict, List, Iterator, Optional
import tempfile
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f

import freesasa
import numpy
import pdb2sql
from scipy.spatial import distance_matrix

<<<<<<< HEAD
from deeprank_gnn.models.error import UnknownAtomError
from deeprank_gnn.tools.pssm import parse_pssm
from deeprank_gnn.tools import BioWrappers, BSA
from deeprank_gnn.tools.pdb import (get_residue_contact_pairs, get_surrounding_residues,
                                    add_hydrogens)
from deeprank_gnn.models.graph import Graph
from deeprank_gnn.domain.graph import EDGETYPE_INTERNAL, EDGETYPE_INTERFACE
from deeprank_gnn.domain.feature import *
from deeprank_gnn.domain.amino_acid import *
from deeprank_gnn.domain.forcefield import atomic_forcefield
from deeprank_gnn.tools.forcefield.potentials import get_coulomb_potentials, get_lennard_jones_potentials
from deeprank_gnn.models.forcefield.vanderwaals import VanderwaalsParam
=======
from deeprank_gnn.domain.amino_acid import *
from deeprank_gnn.domain.feature import *
from deeprank_gnn.models.graph import Graph, Edge, Node
from deeprank_gnn.models.structure import Residue, Atom
from deeprank_gnn.tools.pdb import (get_residue_contact_pairs, get_surrounding_residues,
                                    add_hydrogens)
from deeprank_gnn.tools.pssm import parse_pssm
from deeprank_gnn.tools.graph import build_residue_graph, build_atomic_graph
from deeprank_gnn.models.variant import SingleResidueVariant
import deeprank_gnn.feature.amino_acid
import deeprank_gnn.feature.atomic_contact
import deeprank_gnn.feature.biopython
import deeprank_gnn.feature.bsa
import deeprank_gnn.feature.pssm
import deeprank_gnn.feature.sasa

>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f

_log = logging.getLogger(__name__)


class Query:
    """ Represents one entity of interest, like a single residue variant or a protein-protein interface.

        Query objects are used to generate graphs from structures.
        objects of this class should be created before any model is loaded
    """

    def __init__(self, model_id: str, targets: Optional[Dict[str, float]] = None):

        """
            Args:
                model_id: the id of the model to load, usually a pdb accession code
                targets: target values associated with this query
                pssm_paths: the paths of the pssm files, per protein(chain) id
        """

        self._model_id = model_id

        if targets is None:
            self._targets = {}
        else:
            self._targets = targets

    def _set_graph_targets(self, graph: Graph):
        "simply copies target data from query to graph"

        for target_name, target_data in self._targets.items():
            graph.targets[target_name] = target_data

    def _load_structure(self, pdb_path: str, pssm_paths: Optional[Dict[str, str]] = None):
        "A helper function, to build the structure from pdb and pssm files."

        # make a copy of the pdb, with hydrogens
        pdb_name = os.path.basename(pdb_path)
        hydrogen_pdb_file, hydrogen_pdb_path = tempfile.mkstemp(prefix="hydrogenated-", suffix=pdb_name)
        os.close(hydrogen_pdb_file)

        add_hydrogens(pdb_path, hydrogen_pdb_path)

        # read the pdb copy
        try:
            pdb = pdb2sql.pdb2sql(hydrogen_pdb_path)
        finally:
            os.remove(hydrogen_pdb_path)

        try:
            structure = get_structure(pdb, self.model_id)
        finally:
            pdb._close()

        # read the pssm
        if pssm_paths is not None:
            for chain in structure.chains:
                if chain.id in pssm_paths:
                    pssm_path = pssm_paths[chain.id]

                    with open(pssm_path, 'rt') as f:
                        chain.pssm = parse_pssm(f, chain)

        return structure

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def targets(self) -> Dict[str, float]:
        return self._targets

    def __repr__(self) -> str:
        return "{}({})".format(type(self), self.get_query_id())


class SingleResidueVariantResidueQuery(Query):
    "creates a residue graph from a single residue variant in a pdb file"

<<<<<<< HEAD
    def __init__(self, pdb_path, chain_id, residue_number, insertion_code, wildtype_amino_acid, variant_amino_acid,
                 pssm_paths=None,
                 radius=10.0, external_distance_cutoff=4.5, targets=None, variant_only_features=None):
=======
    def __init__(self, pdb_path: str, chain_id: str, residue_number: int, insertion_code: str,
                 wildtype_amino_acid: AminoAcid, variant_amino_acid: AminoAcid,
                 pssm_paths: Optional[Dict[str, str]] = None, wildtype_conservation: Optional[float] = None,
                 variant_conservation: Optional[float] = None, radius: Optional[float] = 10.0,
                 external_distance_cutoff: Optional[float] = 4.5, targets: Optional[Dict[str, float]] = None):
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f

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

        if variant_only_features is not None:
            self._variant_only_features = variant_only_features
        else:
            self._variant_only_features = {}

    @property
    def residue_id(self) -> str:
        "residue identifier within chain"

        if self._insertion_code is not None:

            return "{}{}".format(self._residue_number, self._insertion_code)
        else:
            return str(self._residue_number)

    def get_query_id(self) -> str:
        return "residue-graph-{}:{}:{}:{}->{}".format(self.model_id, self._chain_id, self.residue_id, self._wildtype_amino_acid.name, self._variant_amino_acid.name)

<<<<<<< HEAD
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
            graph.nodes[node_name][FEATURENAME_HBDONORS] = residue.amino_acid.count_hydrogen_bond_donors
            graph.nodes[node_name][FEATURENAME_HBACCEPTORS] = residue.amino_acid.count_hydrogen_bond_acceptors

            if residue == variant_residue:

                graph.nodes[node_name][FEATURENAME_AMINOACID] = wildtype_amino_acid.onehot
                graph.nodes[node_name][FEATURENAME_VARIANTAMINOACID] = variant_amino_acid.onehot
                graph.nodes[node_name][FEATURENAME_SIZEDIFFERENCE] = variant_amino_acid.size - wildtype_amino_acid.size
                graph.nodes[node_name][FEATURENAME_POLARITYDIFFERENCE] = variant_amino_acid.polarity.onehot - wildtype_amino_acid.polarity.onehot
                graph.nodes[node_name][FEATURENAME_HBDONORSDIFFERENCE] = variant_amino_acid.count_hydrogen_bond_donors - wildtype_amino_acid.count_hydrogen_bond_donors
                graph.nodes[node_name][FEATURENAME_HBACCEPTORSDIFFERENCE] = variant_amino_acid.count_hydrogen_bond_acceptors - wildtype_amino_acid.count_hydrogen_bond_acceptors
            else:
                graph.nodes[node_name][FEATURENAME_AMINOACID] = residue.amino_acid.onehot
                graph.nodes[node_name][FEATURENAME_VARIANTAMINOACID] = numpy.zeros(len(residue.amino_acid.onehot))
                graph.nodes[node_name][FEATURENAME_SIZEDIFFERENCE] = 0
                graph.nodes[node_name][FEATURENAME_POLARITYDIFFERENCE] = numpy.zeros(len(residue.amino_acid.polarity.onehot))
                graph.nodes[node_name][FEATURENAME_HBDONORSDIFFERENCE] = 0
                graph.nodes[node_name][FEATURENAME_HBACCEPTORSDIFFERENCE] = 0

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
        hydrogens_pdb_handle, hydrogens_pdb_path = mkstemp(suffix=".pdb")
        os.close(hydrogens_pdb_handle)

        # get the residues and atoms involved
        try:
            add_hydrogens(self._pdb_path, hydrogens_pdb_path)
            residues = get_surrounding_residues(hydrogens_pdb_path, self.model_id, self._chain_id,
                                                self._residue_number, self._insertion_code,
                                                self._radius)
        finally:
            os.remove(hydrogens_pdb_path)

        structure = list(residues)[0].chain.model

        # read the pssm
        if self._pssm_paths is not None:
            for chain in structure.chains:
                if chain.id in self._pssm_paths:
                    pssm_path = self._pssm_paths[chain.id]
=======
    def build_graph(self, feature_modules: List) -> Graph:
        """ Builds the graph from the pdb structure.
            Args:
                feature_modules (list of modules): each must implement the add_features function.
        """
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths)

        # find the variant residue
        variant_residue = None
<<<<<<< HEAD
        for residue in residues:
            if residue.chain.id == self._chain_id and residue.number == self._residue_number and residue.insertion_code == self._insertion_code:
                variant_residue = residue
                break
        else:
            raise ValueError("Residue {}:{} not found in {}".format(self._chain_id, self.residue_id, self._pdb_path))

        atoms = set([])
        for residue in residues:
            if residue.amino_acid is not None:
                for atom in residue.atoms:
                    atoms.add(atom)
        atoms = list(atoms)

        # build a graph and keep track of how we named the nodes
        node_name_residues = {}
        graph = Graph(self.get_query_id(), self.targets)

        # find neighbouring atoms
        atom_positions = [atom.position for atom in atoms]
        interatomic_distances = distance_matrix(atom_positions, atom_positions, p=2)
        neighbours = interatomic_distances < self._external_distance_cutoff
=======
        for residue in structure.get_chain(self._chain_id).residues:
            if residue.number == self._residue_number and residue.insertion_code == self._insertion_code:
                variant_residue = residue
                break
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f

        if variant_residue is None:
            raise ValueError("Residue not found in {}: {} {}"
                             .format(self._pdb_path, self._chain_id, self.residue_id))

<<<<<<< HEAD
                atom_distance = interatomic_distances[atom1_index, atom2_index]
=======
        # define the variant
        variant = SingleResidueVariant(variant_residue, self._variant_amino_acid)
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f

        # select which residues will be the graph
        residues = get_surrounding_residues(structure, residue, self._radius)

        # build the graph
        graph = build_residue_graph(residues, self.get_query_id(), self._external_distance_cutoff)

        # add data to the graph
        self._set_graph_targets(graph)

<<<<<<< HEAD
                    residue1_key = self._get_residue_node_key(residue1)
                    residue2_key = self._get_residue_node_key(residue2)

                    node_name_residues[residue1_key] = residue1
                    node_name_residues[residue2_key] = residue2

                    # add the edge if not already
                    if not graph.has_edge(residue1_key, residue2_key):
                        graph.add_edge(residue1_key, residue2_key)
                        graph.edges[residue1_key, residue2_key][FEATURENAME_EDGETYPE] = EDGETYPE_INTERFACE

                    # covalent bond overrides non-covalent
                    if self._is_covalent_bond(atom1, atom2, atom_distance):
                        graph.edges[residue1_key, residue2_key][FEATURENAME_EDGETYPE] = EDGETYPE_INTERNAL

                    # Make sure we take the shortest distance
                    if FEATURENAME_EDGEDISTANCE not in graph.edges[residue1_key, residue2_key]:
                        graph.edges[residue1_key, residue2_key][FEATURENAME_EDGEDISTANCE] = atom_distance

                    elif graph.edges[residue1_key, residue2_key][FEATURENAME_EDGEDISTANCE] > atom_distance:
                        graph.edges[residue1_key, residue2_key][FEATURENAME_EDGEDISTANCE] = atom_distance

        # set the node features
        self._set_amino_acid_properties(graph, node_name_residues, variant_residue, self._wildtype_amino_acid, self._variant_amino_acid)
        self._set_sasa(graph, node_name_residues, self._pdb_path)

        if self._pssm_paths is not None:
            self._set_pssm(graph, node_name_residues, variant_residue,
                           self._wildtype_amino_acid, self._variant_amino_acid)

        self._set_contacts(graph, node_name_residues, atoms, interatomic_distances)

        # set the variant-only features
        for feature_name, feature_value in self._variant_only_features.items():
            zero_value = self._get_zero_value(feature_value)
            for node_name, node in graph.nodes.items():
                residue = node_name_residues[node_name]
                if residue == variant_residue:
                    node[feature_name] = feature_value
                else:
                    node[feature_name] = zero_value
=======
        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph, variant)
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f

        return graph

    @staticmethod
    def _get_zero_value(value):
        if type(value) == float or type(value) == numpy.float32 or type(value) == numpy.float64:
            return 0.0

        elif type(value) == int:
            return 0

        elif isinstance(value, numpy.ndarray):
            return numpy.zeros(value.shape, dtype=float)

        else:
            raise TypeError(type(value))

    @staticmethod
    def _set_contacts(graph, node_name_residues, atoms, interatomic_distances):

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

        # calculate potentials
        interatomic_electrostatic_potentials = get_coulomb_potentials(interatomic_distances, atom_charges)
        interatomic_vanderwaals_potentials = get_lennard_jones_potentials(interatomic_distances, atoms, atom_vanderwaals_parameters)

        # set the features
        for edge_key, edge_features in graph.edges.items():
            node1_name, node2_name = edge_key

            residue1 = node_name_residues[node1_name]
            residue2 = node_name_residues[node2_name]

            for atom1 in residue1.atoms:
                for atom2 in residue2.atoms:

                    atom1_index = atom_indices[atom1]
                    atom2_index = atom_indices[atom2]

                    edge_features[FEATURENAME_EDGEVANDERWAALS] = (edge_features.get(FEATURENAME_EDGEVANDERWAALS, 0.0) +
                                                                  interatomic_vanderwaals_potentials[atom1_index, atom2_index])

                    edge_features[FEATURENAME_EDGECOULOMB] = (edge_features.get(FEATURENAME_EDGECOULOMB, 0.0) +
                                                              interatomic_electrostatic_potentials[atom1_index, atom2_index])

class SingleResidueVariantAtomicQuery(Query):
    "creates an atomic graph for a single residue variant in a pdb file"

<<<<<<< HEAD
    def __init__(self, pdb_path, chain_id, residue_number, insertion_code, wildtype_amino_acid, variant_amino_acid,
                 pssm_paths=None,
                 radius=10.0, external_distance_cutoff=4.5, internal_distance_cutoff=3.0,
                 targets=None):
=======
    def __init__(self, pdb_path: str, chain_id: str, residue_number: int, insertion_code: str,
                 wildtype_amino_acid: AminoAcid, variant_amino_acid: AminoAcid,
                 pssm_paths: Optional[Dict[str, str]] = None,
                 radius: Optional[float] = 10.0,
                 external_distance_cutoff: Optional[float] = 4.5,
                 targets: Optional[Dict[str, float]] = None):
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f
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
    def residue_id(self) -> str:
        "string representation of the residue number and insertion code"

        if self._insertion_code is not None:
            return "{}{}".format(self._residue_number, self._insertion_code)
        else:
            return str(self._residue_number)

    def get_query_id(self) -> str:
        return "{}:{}:{}:{}->{}".format(self.model_id, self._chain_id, self.residue_id, self._wildtype_amino_acid.name, self._variant_amino_acid.name)

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.model_id == other.model_id and \
            self._chain_id == other._chain_id and self.residue_id == other.residue_id and \
            self._wildtype_amino_acid == other._wildtype_amino_acid and self._variant_amino_acid == other._variant_amino_acid

    def __hash__(self) -> hash:
        return hash((self.model_id, self._chain_id, self.residue_id, self._wildtype_amino_acid, self._variant_amino_acid))

    @staticmethod
    def _get_atom_node_key(atom) -> str:
        """ Pickle has problems serializing the graph when the nodes are atoms,
            so use this function to generate an unique key for the atom"""

        # This should include the model, chain, residue and atom
        return str(atom)

<<<<<<< HEAD
    def build_graph(self):
        hydrogens_pdb_handle, hydrogens_pdb_path = mkstemp(suffix=".pdb")
        os.close(hydrogens_pdb_handle)

        try:
            add_hydrogens(self._pdb_path, hydrogens_pdb_path)

            # get the residues and atoms involved
            residues = get_surrounding_residues(hydrogens_pdb_path, self.model_id, self._chain_id,
                                                self._residue_number, self._insertion_code,
                                                self._radius)
        finally:
            os.remove(hydrogens_pdb_path)

        structure = list(residues)[0].chain.model
=======
    def build_graph(self, feature_modules: List) -> Graph:
        """ Builds the graph from the pdb structure.
            Args:
                feature_modules (list of modules): each must implement the add_features function.
        """

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths)
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f

        # find the variant residue
        variant_residue = None
        for residue in structure.get_chain(self._chain_id).residues:
            if residue.number == self._residue_number and residue.insertion_code == self._insertion_code:
                variant_residue = residue
                break

        if variant_residue is None:
            raise ValueError("Residue not found in {}: {} {}"
                             .format(self._pdb_path, self._chain_id, self.residue_id))

<<<<<<< HEAD
        # find the variant residue
        variant_residue = None
        for residue in residues:
            if residue.chain.id == self._chain_id and residue.number == self._residue_number and residue.insertion_code == self._insertion_code:
                variant_residue = residue
                break
        else:
            raise ValueError("Residue {}:{} not found in {}".format(self._chain_id, self.residue_id, self._pdb_path))

=======
        # define the variant
        variant = SingleResidueVariant(variant_residue, self._variant_amino_acid)

        # get the residues and atoms involved
        residues = get_surrounding_residues(structure, variant_residue, self._radius)
        residues.add(variant_residue)
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f
        atoms = set([])
        for residue in residues:
            if residue.amino_acid is not None:
                for atom in residue.atoms:
                    atoms.add(atom)
        atoms = list(atoms)
<<<<<<< HEAD

        # build a graph and keep track of how we named the nodes
        node_name_atoms = {}
        graph = Graph(self.get_query_id(), self.targets)

        # find neighbouring atoms
        atom_positions = [atom.position for atom in atoms]
        interatomic_distances = distance_matrix(atom_positions, atom_positions, p=2)
        neighbours = interatomic_distances < self._external_distance_cutoff
=======

        # build the graph
        graph = build_atomic_graph(atoms, self.get_query_id(), self._external_distance_cutoff)
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f

        # add data to the graph
        self._set_graph_targets(graph)

<<<<<<< HEAD
        # give every chain a code
        for chain in structure.chains:
            chain_codes[chain] = len(chain_codes)

        # iterate over every pair of neighbouring atoms
        for atom1_index, atom2_index in numpy.transpose(numpy.nonzero(neighbours)):
            if atom1_index != atom2_index:  # do not pair an atom with itself

                distance = interatomic_distances[atom1_index, atom2_index]

                atom1 = atoms[atom1_index]
                atom2 = atoms[atom2_index]

                atom1_key = SingleResidueVariantAtomicQuery._get_atom_node_key(atom1)
                atom2_key = SingleResidueVariantAtomicQuery._get_atom_node_key(atom2)

                # connect the atoms and set the distance
                graph.add_edge(atom1_key, atom2_key)

                if distance < self._internal_distance_cutoff:
                    graph.edges[atom1_key, atom2_key][FEATURENAME_EDGETYPE] = EDGETYPE_INTERNAL
                else:
                    graph.edges[atom1_key, atom2_key][FEATURENAME_EDGETYPE] = EDGETYPE_INTERFACE

                graph.edges[atom1_key, atom2_key][FEATURENAME_EDGEDISTANCE] = distance

                # set the positions of the atoms
                graph.nodes[atom1_key][FEATURENAME_POSITION] = atom1.position
                graph.nodes[atom2_key][FEATURENAME_POSITION] = atom2.position
                graph.nodes[atom1_key][FEATURENAME_CHAIN] = chain_codes[atom1.residue.chain]
                graph.nodes[atom2_key][FEATURENAME_CHAIN] = chain_codes[atom2.residue.chain]

                node_name_atoms[atom1_key] = atom1
                node_name_atoms[atom2_key] = atom2

        # set additional features
        SingleResidueVariantAtomicQuery._set_contacts(graph, node_name_atoms, atoms, interatomic_distances)

        SingleResidueVariantAtomicQuery._set_pssm(graph, node_name_atoms, variant_residue,
                                                  self._wildtype_amino_acid, self._variant_amino_acid)

        SingleResidueVariantAtomicQuery._set_sasa(graph, node_name_atoms, self._pdb_path)

        SingleResidueVariantAtomicQuery._set_amino_acid(graph, node_name_atoms, variant_residue, self._wildtype_amino_acid, self._variant_amino_acid)
=======
        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph, variant)
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f

        return graph


class ProteinProteinInterfaceAtomicQuery(Query):
    "a query that builds atom-based graphs, using the residues at a protein-protein interface"

<<<<<<< HEAD
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
    def _set_contacts(graph, node_name_atoms, atoms, interatomic_distances):

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

        # calculate potentials
        interatomic_electrostatic_potentials = get_coulomb_potentials(interatomic_distances, atom_charges)
        interatomic_vanderwaals_potentials = get_lennard_jones_potentials(interatomic_distances, atoms, atom_vanderwaals_parameters)

        # set the features
        for edge_key, edge_features in graph.edges.items():
            node1_name, node2_name = edge_key

            atom1 = node_name_atoms[node1_name]
            atom2 = node_name_atoms[node2_name]

            atom1_index = atom_indices[atom1]
            atom2_index = atom_indices[atom2]

            edge_features[FEATURENAME_EDGEDISTANCE] = interatomic_distances[atom1_index, atom2_index]

            edge_features[FEATURENAME_EDGEVANDERWAALS] = interatomic_vanderwaals_potentials[atom1_index, atom2_index]

            edge_features[FEATURENAME_EDGECOULOMB] = interatomic_electrostatic_potentials[atom1_index, atom2_index]
=======
    def __init__(self, pdb_path: str, chain_id1: str, chain_id2: str, pssm_paths: Optional[Dict[str, str]] = None,
                 interface_distance_cutoff: Optional[float] = 8.5,
                 targets: Optional[Dict[str, float]] = None):
        """
            Args:
                pdb_path(str): the path to the pdb file
                chain_id1(str): the pdb chain identifier of the first protein of interest
                chain_id2(str): the pdb chain identifier of the second protein of interest
                pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
                interface_distance_cutoff(float): max distance in Ångström between two interacting residues of the two proteins
                targets(dict, optional): named target values associated with this query
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._interface_distance_cutoff = interface_distance_cutoff

    def get_query_id(self) -> str:
        return "atom-ppi-{}:{}-{}".format(self.model_id, self._chain_id1, self._chain_id2)

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.model_id == other.model_id and \
            {self._chain_id1, self._chain_id2} == {other._chain_id1, other._chain_id2}

    def __hash__(self) -> hash:
        return hash((self.model_id, tuple(sorted([self._chain_id1, self._chain_id2]))))

    def build_graph(self, feature_modules: List) -> Graph:
        """ Builds the graph from the pdb structure.
            Args:
                feature_modules (list of modules): each must implement the add_features function.
        """

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths)

        # get the contact residues
        interface_pairs = get_residue_contact_pairs(self._pdb_path, structure,
                                                    self._chain_id1, self._chain_id2,
                                                    self._interface_distance_cutoff)
        if len(interface_pairs) == 0:
            raise ValueError("no interface residues found")

        atoms_selected = set([])
        for residue1, residue2 in interface_pairs:
            for atom in (residue1.atoms + residue2.atoms):
                atoms_selected.add(atom)
        atoms_selected = list(atoms_selected)

        # build the graph
        graph = build_atomic_graph(atoms_selected, self.get_query_id(), self._interface_distance_cutoff)

        # add data to the graph
        self._set_graph_targets(graph)

        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph)

        return graph
>>>>>>> 611c2cbf5d47840fa15ce7c3c8b8bcfe7de5d55f


class ProteinProteinInterfaceResidueQuery(Query):
    "a query that builds residue-based graphs, using the residues at a protein-protein interface"

    def __init__(self, pdb_path: str, chain_id1: str, chain_id2: str, pssm_paths: Optional[Dict[str, str]] = None,
                 interface_distance_cutoff: float = 8.5,
                 targets: Optional[Dict[str, float]] = None):
        """
            Args:
                pdb_path(str): the path to the pdb file
                chain_id1(str): the pdb chain identifier of the first protein of interest
                chain_id2(str): the pdb chain identifier of the second protein of interest
                pssm_paths(dict(str,str), optional): the paths to the pssm files, per chain identifier
                interface_distance_cutoff(float): max distance in Ångström between two interacting residues of the two proteins
                targets(dict, optional): named target values associated with this query
        """

        model_id = os.path.splitext(os.path.basename(pdb_path))[0]

        Query.__init__(self, model_id, targets)

        self._pdb_path = pdb_path

        self._chain_id1 = chain_id1
        self._chain_id2 = chain_id2

        self._pssm_paths = pssm_paths

        self._interface_distance_cutoff = interface_distance_cutoff

    def get_query_id(self) -> str:
        return "residue-ppi-{}:{}-{}".format(self.model_id, self._chain_id1, self._chain_id2)

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self.model_id == other.model_id and \
            {self._chain_id1, self._chain_id2} == {other._chain_id1, other._chain_id2}

    def __hash__(self) -> hash:
        return hash((self.model_id, tuple(sorted([self._chain_id1, self._chain_id2]))))

    def build_graph(self, feature_modules: List) -> Graph:
        """ Builds the graph from the pdb structure.
            Args:
                feature_modules (list of modules): each must implement the add_features function.
        """

        # load pdb structure
        structure = self._load_structure(self._pdb_path, self._pssm_paths)

        # get the contact residues
        interface_pairs = get_residue_contact_pairs(self._pdb_path, structure,
                                                    self._chain_id1, self._chain_id2,
                                                    self._interface_distance_cutoff)

        if len(interface_pairs) == 0:
            raise ValueError("no interface residues found")

        residues_selected = set([])
        for residue1, residue2 in interface_pairs:
            residues_selected.add(residue1)
            residues_selected.add(residue2)
        residues_selected = list(residues_selected)

        # build the graph
        graph = build_residue_graph(residues_selected, self.get_query_id(), self._interface_distance_cutoff)

        # add data to the graph
        self._set_graph_targets(graph)

        for feature_module in feature_modules:
            feature_module.add_features(self._pdb_path, graph)

        return graph


class QueryDataset:
    "represents a collection of data queries"

    def __init__(self):
        self._queries = []

    def add(self, query: Query):
        self._queries.append(query)

    @property
    def queries(self) -> List[Query]:
        return self._queries

    def __contains__(self, query: Query) -> bool:
        return query in self._queries

    def __iter__(self) -> Iterator[Query]:
        return iter(self._queries)
