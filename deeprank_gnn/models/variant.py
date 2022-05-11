
from deeprank_gnn.models.amino_acid import AminoAcid
from deeprank_gnn.models.structure import Residue


class SingleResidueVariant:
    "represents an amino acid replacement"

    def __init__(self, residue: Residue, variant_amino_acid: AminoAcid):
        self._residue = residue
        self._variant_amino_acid = variant_amino_acid

    @property
    def residue(self) -> Residue:
        return self._residue

    @property
    def variant_amino_acid(self) -> AminoAcid:
        return self._variant_amino_acid

    @property
    def wildtype_amino_acid(self) -> AminoAcid:
        return self._residue.amino_acid
