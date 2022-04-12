import numpy

from deeprank_gnn.models.polarity import Polarity


class AminoAcid:
    "a value to represent one of the amino acids"

    def __init__(self, name: str, three_letter_code: str, one_letter_code: str,
                 charge: float, polarity: Polarity, size: int,
                 count_hydrogen_bond_donors: int, count_hydrogen_bond_acceptors: int,
                 index: int):

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
            raise ValueError("amino acid {} index is not set, thus no onehot can be computed".format(self._name))

        a = numpy.zeros(20)  # assumed that there are only 20 different amino acids
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
        return type(other) == type(self) and other.name == self.name

    def __repr__(self):
        return self._name
