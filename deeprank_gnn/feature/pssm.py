from typing import Optional

import numpy

from deeprank_gnn.models.variant import SingleResidueVariant
from deeprank_gnn.domain.amino_acid import amino_acids
from deeprank_gnn.models.structure import Residue, Atom
from deeprank_gnn.models.graph import Graph
from deeprank_gnn.domain.feature import (FEATURENAME_PSSM, FEATURENAME_PSSMDIFFERENCE,
                                         FEATURENAME_PSSMWILDTYPE, FEATURENAME_PSSMVARIANT,
                                         FEATURENAME_INFORMATIONCONTENT)

profile_amino_acid_order = sorted(amino_acids, key=lambda aa: aa.one_letter_code)


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

        pssm_row = residue.get_pssm()

        profile = numpy.array([pssm_row.get_conservation(amino_acid)
                               for amino_acid in profile_amino_acid_order])

        node.features[FEATURENAME_PSSM] = profile
        node.features[FEATURENAME_INFORMATIONCONTENT] = pssm_row.information_content

        if single_amino_acid_variant is not None:

            if residue == single_amino_acid_variant.residue:
                # only the variant residue can have a variant and wildtype amino acid

                conservation_wildtype = pssm_row.get_conservation(single_amino_acid_variant.wildtype_amino_acid)
                conservation_variant = pssm_row.get_conservation(single_amino_acid_variant.variant_amino_acid)

                node.features[FEATURENAME_PSSMWILDTYPE] = conservation_wildtype
                node.features[FEATURENAME_PSSMVARIANT] = conservation_variant
                node.features[FEATURENAME_PSSMDIFFERENCE] = conservation_variant - conservation_wildtype
            else:
                # all nodes must have the same features, so set them to zero here
                node.features[FEATURENAME_PSSMWILDTYPE] = 0.0
                node.features[FEATURENAME_PSSMVARIANT] = 0.0
                node.features[FEATURENAME_PSSMDIFFERENCE] = 0.0


