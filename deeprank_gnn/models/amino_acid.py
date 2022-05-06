import numpy

from .polarity import Polarity


class AminoAcid:
    "a value to represent one of the amino acids"

    def __init__( # pylint: disable=too-many-arguments
        self,
        name: str,
        three_letter_code: str,
        one_letter_code: str,
        charge: float,
        polarity: Polarity,
        size: int,
        count_hydrogen_bond_donors: int,
        count_hydrogen_bond_acceptors: int,
        index: int,
    ):
        """
        Args:
            name(str): unique name for the amino acid
            three_letter_code(str): code of the amino acid, as in PDB
            one_letter_code(str): letter of the amino acid, as in fasta
            charge(float, optional): the charge property of the amino acid
            polarity(deeprank polarity enum, optional): the polarity property of the amino acid
            size(int, optional): the number of heavy atoms in the side chain
            index(int, optional): the rank of the amino acid, used for computing one-hot encoding
        """

        self._name = name
        self._three_letter_code = three_letter_code
        self._one_letter_code = one_letter_code

        # these settings apply to the side chain
        self._size = size
        self._charge = charge
        self._polarity = polarity
        self._count_hydrogen_bond_donors = count_hydrogen_bond_donors
        self._count_hydrogen_bond_acceptors = count_hydrogen_bond_acceptors

        self._index = index

    @property
    def name(self) -> str:
        return self._name

    @property
    def three_letter_code(self) -> str:
        return self._three_letter_code

    @property
    def one_letter_code(self) -> str:
        return self._one_letter_code

    @property
    def onehot(self) -> numpy.ndarray:
        if self._index is None:
            raise ValueError(
                "amino acid {self._name} index is not set, thus no onehot can be computed"
            )

        # assumed that there are only 20 different amino acids
        a = numpy.zeros(20)
        a[self._index] = 1.0

        return a

    @property
    def count_hydrogen_bond_donors(self) -> int:
        return self._count_hydrogen_bond_donors

    @property
    def count_hydrogen_bond_acceptors(self) -> int:
        return self._count_hydrogen_bond_acceptors

    @property
    def charge(self) -> float:
        return self._charge

    @property
    def polarity(self) -> Polarity:
        return self._polarity

    @property
    def size(self) -> int:
        return self._size

    @property
    def index(self) -> int:
        return self._index

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, type(self)) and other.name == self.name

    def __repr__(self):
        return self._name
