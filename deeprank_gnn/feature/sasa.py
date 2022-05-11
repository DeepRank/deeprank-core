from typing import List
import numpy
import freesasa
from deeprank_gnn.models.structure import Residue, Atom
from deeprank_gnn.models.graph import Node, Graph
from deeprank_gnn.domain.feature import FEATURENAME_SASA


def add_features_for_residues(structure: freesasa.Structure, # pylint: disable=c-extension-no-member
                              result: freesasa.Result, # pylint: disable=c-extension-no-member
                              nodes: List[Node]):

    for node in nodes:

        residue = node.id

        selection = ("residue, (resi %s) and (chain %s)" % (residue.number_string, residue.chain.id),)

        area = freesasa.selectArea(selection, structure, result)['residue'] # pylint: disable=c-extension-no-member
        if numpy.isnan(area):
            raise ValueError(f"freesasa returned {area} for {residue}")

        node.features[FEATURENAME_SASA] = area


def add_features_for_atoms(structure: freesasa.Structure, # pylint: disable=c-extension-no-member
                           result: freesasa.Result, # pylint: disable=c-extension-no-member
                           nodes: List[Node]):

    for node in nodes:

        atom = node.id

        selection = ('atom, (name %s) and (resi %s) and (chain %s)' % (atom.name, atom.residue.number_string, atom.residue.chain.id),)

        area = freesasa.selectArea(selection, structure, result)['atom'] # pylint: disable=c-extension-no-member
        if numpy.isnan(area):
            raise ValueError(f"freesasa returned {area} for {atom}")

        node.features[FEATURENAME_SASA] = area


def add_features(pdb_path: str, graph: Graph, *args, **kwargs): # pylint: disable=unused-argument

    structure = freesasa.Structure(pdb_path) # pylint: disable=c-extension-no-member
    result = freesasa.calc(structure) # pylint: disable=c-extension-no-member

    if type(graph.nodes[0].id) == Residue:
        return add_features_for_residues(structure, result, graph.nodes)

    elif type(graph.nodes[0].id) == Atom:
        return add_features_for_atoms(structure, result, graph.nodes)
    else:
        raise TypeError("Unexpected node type: {}".format(type(nodes[0].id)))