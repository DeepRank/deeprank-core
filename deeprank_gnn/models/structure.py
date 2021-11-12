from enum import Enum


class AtomicElement(Enum):
    C = 1
    O = 2
    N = 3
    S = 4
    P = 5
    H = 6


class Atom:
    def __init__(self, residue, name, element, position):
        self._residue = residue
        self._name = name
        self._element = element
        self._position = position

    def __eq__(self, other):
        return type(self) == type(other) and \
            self._residue == other._residue and \
            self._name == other._name

    def __hash__(self):
        return hash((self._residue, self._name))

    @property
    def name(self):
        return self._name

    @property
    def element(self):
        return self._element

    @property
    def position(self):
        return self._position

    @property
    def residue(self):
        return self._residue


class Residue:
    def __init__(self, chain, number, amino_acid=None, insertion_code=None):
        self._chain = chain
        self._number = number
        self._amino_acid = amino_acid
        self._insertion_code = insertion_code
        self._atoms = []

    def __eq__(self, other):
        return type(self) == type(other) and \
            self._chain == other._chain and \
            self._number == other._number and \
            self._insertion_code == other._insertion_code

    def __hash__(self):
        return hash((self._chain, self._number, self._insertion_code))

    @property
    def number(self):
        return self._number

    @property
    def chain(self):
        return self._chain

    @property
    def amino_acid(self):
        return self._amino_acid

    @property
    def atoms(self):
        return self._atoms

    @property
    def number_string(self):
        "contains both the number and the insertion code (if any)"

        if self._insertion_code is not None:
            return "{}{}".format(self._number, self._insertion_code)
        else:
            return str(self._number)

    @property
    def atoms(self):
        return self._atoms

    def add_atom(self, atom):
        self._atoms.append(atom)

    def __repr__(self):
        return "{} {}".format(self._chain, self.get_number_string())


class Chain:
    def __init__(self, model, id_):
        self._model = model
        self._id = id_
        self._residues = []

    def add_residue(self, residue):
        self._residues.append(residue)

    @property
    def id(self):
        return self._id

    @property
    def residues(self):
        return self._residues

    def __eq__(self, other):
        return type(self) == type(other) and self._model == other._model and self._id == other._id

    def __hash__(self):
        return hash((self._model, self._id))

    def __repr__(self):
        return "{} {}".format(self._model, self._id)


class Structure:
    def __init__(self, id_):
        self._id = id_
        self._chains = []

    def __eq__(self, other):
        return type(self) == type(other) and self._id == other._id

    def __hash__(self):
        return hash(self._id)

    def __repr__(self):
        return self._id

    def add_chain(self, chain):
        self._chains.append(chain)

    @property
    def chains(self):
        return self._chains