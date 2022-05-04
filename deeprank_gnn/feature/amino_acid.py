from typing import Optional

import numpy

from deeprank_gnn.models.graph import Graph
from deeprank_gnn.models.variant import SingleResidueVariant
from deeprank_gnn.models.structure import Atom, Residue
from deeprank_gnn.domain.feature import (FEATURENAME_AMINOACID, FEATURENAME_VARIANTAMINOACID,
                                         FEATURENAME_SIZE, FEATURENAME_POLARITY,
                                         FEATURENAME_SIZEDIFFERENCE, FEATURENAME_POLARITYDIFFERENCE,
                                         FEATURENAME_HYDROGENBONDDONORS, FEATURENAME_HYDROGENBONDDONORSDIFFERENCE,
                                         FEATURENAME_HYDROGENBONDACCEPTORS, FEATURENAME_HYDROGENBONDACCEPTORSDIFFERENCE)

def add_features(pdb_path: str, graph: Graph,
                 single_amino_acid_variant: Optional[SingleResidueVariant] = None):

    for node in graph.nodes:
        if type(node.id) == Residue:
            residue = node.id

        elif type(node.id) == Atom:
            atom = node.id
            residue = atom.residue
        else:
            raise TypeError("Unexpected node type: {}".format(type(node.id))) 

        node.features[FEATURENAME_AMINOACID] = residue.amino_acid.onehot
        node.features[FEATURENAME_SIZE] = residue.amino_acid.size
        node.features[FEATURENAME_POLARITY] = residue.amino_acid.polarity.onehot
        node.features[FEATURENAME_HYDROGENBONDDONORS] = residue.amino_acid.count_hydrogen_bond_donors
        node.features[FEATURENAME_HYDROGENBONDACCEPTORS] = residue.amino_acid.count_hydrogen_bond_acceptors

        if single_amino_acid_variant is not None:

            wildtype = single_amino_acid_variant.wildtype_amino_acid
            variant = single_amino_acid_variant.variant_amino_acid

            if residue == single_amino_acid_variant.residue:
                node.features[FEATURENAME_SIZEDIFFERENCE] = variant.size - wildtype.size
                node.features[FEATURENAME_VARIANTAMINOACID] = variant.onehot
                node.features[FEATURENAME_POLARITYDIFFERENCE] = variant.polarity.onehot - wildtype.polarity.onehot
                node.features[FEATURENAME_HYDROGENBONDDONORSDIFFERENCE] = variant.count_hydrogen_bond_donors - wildtype.count_hydrogen_bond_donors
                node.features[FEATURENAME_HYDROGENBONDACCEPTORSDIFFERENCE] = variant.count_hydrogen_bond_acceptors - wildtype.count_hydrogen_bond_acceptors
            else:
                node.features[FEATURENAME_SIZEDIFFERENCE] = 0
                node.features[FEATURENAME_VARIANTAMINOACID] = residue.amino_acid.onehot
                node.features[FEATURENAME_POLARITYDIFFERENCE] = numpy.zeros(residue.amino_acid.polarity.onehot.shape)
                node.features[FEATURENAME_HYDROGENBONDDONORSDIFFERENCE] = 0
                node.features[FEATURENAME_HYDROGENBONDACCEPTORSDIFFERENCE] = 0


