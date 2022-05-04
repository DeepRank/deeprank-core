from typing import Dict, List, Optional
import logging

import numpy
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import ResidueDepth, get_surface, residue_depth
from Bio.PDB.HSExposure import HSExposureCA

from deeprank_gnn.domain.feature import FEATURENAME_RESIDUEDEPTH, FEATURENAME_HALFSPHEREEXPOSURE
from deeprank_gnn.models.structure import Atom, Residue
from deeprank_gnn.models.graph import Graph


_log = logging.getLogger(__name__)


def space_if_none(value):
    if value is None:
        return " "

    return value


def add_features(pdb_path: str, graph: Graph, *args, **kwargs):

    parser = PDBParser()
    structure = parser.get_structure('_tmp', pdb_path)
    bio_model = structure[0]

    surface = get_surface(bio_model)
    hse = HSExposureCA(bio_model)

    for node in graph.nodes:

        if type(node.id) == Atom:
            atom = node.id
            residue = atom.residue

        elif type(node.id) == Residue:
            residue = node.id
        else:
            raise TypeError("Unexpected node type: {}".format(type(node)))

        # These can only be calculated per residue, not per atom.
        # So for atomic graphs, every atom gets its residue's value.
        bio_residue = bio_model[residue.chain.id][residue.number]
        node.features[FEATURENAME_RESIDUEDEPTH] = residue_depth(bio_residue, surface)

        hse_key = (residue.chain.id, (" ", residue.number, space_if_none(residue.insertion_code)))
        if hse_key in hse:
            node.features[FEATURENAME_HALFSPHEREEXPOSURE] = hse[hse_key]
        else:
            node.features[FEATURENAME_HALFSPHEREEXPOSURE] = numpy.array((0, 0, 0))
