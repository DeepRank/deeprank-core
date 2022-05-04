from typing import List, Optional

import numpy

from enum import Enum

from deeprank_gnn.models.amino_acid import AminoAcid
from deeprank_gnn.models.pssm import PssmRow


class Structure:
    "represents one entire pdb structure"

    def __init__(self, id_: Optional[str] = None):
        """
            Args:
                id_(str): an unique identifier for this structure, can be the pdb accession code.
        """
        self._id = id_
        self._chains = {}

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self._id == other._id

    def __hash__(self) -> hash:
        return hash(self._id)

    def __repr__(self) -> str:
        return self._id

    def get_chain(self, chain_id: str):
        return self._chains[chain_id]

    def add_chain(self, chain):
        if chain.id in self._chains:
            raise ValueError("duplicate chain: {}".format(chain.id))

        self._chains[chain.id] = chain

    @property
    def chains(self):
        return list(self._chains.values())

    def get_atoms(self):
        "shortcut to list all atoms in this structure"
        atoms = []
        for chain in self._chains.values():
            for residue in chain.residues:
                atoms.extend(residue.atoms)

        return atoms

    @property
    def id(self) -> str:
        return self._id


class Chain:
    "represents one pdb chain"

    def __init__(self, model: Structure, id_: Optional[str]):
        """
        Args:
            model(deeprank structure object): the model that this chain is part of
            id_(str): the pdb identifier of this chain
        """

        self._model = model
        self._id = id_
        self._residues = []
        self._pssm = None  # pssm is per chain

    @property
    def model(self):
        return self._model

    @property
    def pssm(self) -> PssmRow:
        return self._pssm

    @pssm.setter
    def pssm(self, pssm: PssmRow):
        self._pssm = pssm

    def add_residue(self, residue):
        self._residues.append(residue)

    @property
    def id(self) -> str:
        return self._id

    @property
    def residues(self):
        return self._residues

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and self._model == other._model and self._id == other._id

    def __hash__(self) -> hash:
        return hash((self._model, self._id))

    def __repr__(self) -> str:
        return "{} {}".format(self._model, self._id)


class Residue:
    "represents a pdb residue"

    def __init__(self, chain: Chain, number: int, amino_acid: Optional[AminoAcid] = None,
                 insertion_code: Optional[str] = None):
        """
        Args:
            chain(deeprank chain object): the chain that this residue belongs to
            number(int): the residue number
            amino_acid(deeprank amino acid, optional): the residue's amino acid (if it's part of a protein)
            insertion_code(str, optional): the pdb insertion code, if any
        """

        self._chain = chain
        self._number = number
        self._amino_acid = amino_acid
        self._insertion_code = insertion_code
        self._atoms = []

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and \
            self._chain == other._chain and \
            self._number == other._number and \
            self._insertion_code == other._insertion_code

    def __hash__(self) -> hash:
        return hash((self._chain, self._number, self._insertion_code))

    def get_pssm(self) -> PssmRow:
        """ if the residue's chain has pssm info linked to it,
            then return the part that belongs to this residue
        """

        pssm = self._chain.pssm
        if pssm is None:
            raise ValueError("pssm not set on {}".format(self._chain))

        return pssm[self]

    @property
    def number(self) -> int:
        return self._number

    @property
    def chain(self):
        return self._chain

    @property
    def amino_acid(self) -> AminoAcid:
        return self._amino_acid

    @property
    def atoms(self):
        return self._atoms

    @property
    def number_string(self) -> str:
        "contains both the number and the insertion code (if any)"

        if self._insertion_code is not None:
            return "{}{}".format(self._number, self._insertion_code)
        else:
            return str(self._number)

    @property
    def insertion_code(self) -> str:
        return self._insertion_code

    def add_atom(self, atom):
        self._atoms.append(atom)

    def __repr__(self) -> str:
        return "{} {}".format(self._chain, self.number_string)

    @property
    def position(self) -> numpy.array:
        return numpy.mean([atom.position for atom in self._atoms], axis=0)


class AtomicElement(Enum):
    "value to represent the type of pdb atoms"

    C = 1
    O = 2
    N = 3
    S = 4
    P = 5
    H = 6

    @property
    def onehot(self) -> numpy.array:
        value = numpy.zeros(max([el.value for el in AtomicElement]))
        value[self.value - 1] = 1.0
        return value


class Atom:
    "represents a pdb atom"

    def __init__(self, residue: Residue, name: str, element: AtomicElement, position: numpy.array, occupancy: float):
        """
            Args:
                residue(deeprank residue object): the residue that this atom belongs to
                name(str): pdb atom name
                element(deeprank atomic element enumeration): the chemical element
                position(numpy array of length 3): pdb position xyz of this atom
                occupancy(float): pdb occupancy value
        """
        self._residue = residue
        self._name = name
        self._element = element
        self._position = position
        self._occupancy = occupancy

    def __eq__(self, other) -> bool:
        return type(self) == type(other) and \
            self._residue == other._residue and \
            self._name == other._name

    def __hash__(self) -> hash:
        return hash((self._residue, self._name))

    def __repr__(self) -> str:
        return "{} {}".format(self._residue, self._name)

    def change_altloc(self, alternative_atom):
        "replace the atom's location by another atom's location"

        self._position = alternative_atom.position
        self._occupancy = alternative_atom.occupancy

    @property
    def name(self) -> str:
        return self._name

    @property
    def element(self) -> AtomicElement:
        return self._element

    @property
    def occupancy(self) -> float:
        return self._occupancy

    @property
    def position(self) -> numpy.array:
        return self._position

    @property
    def residue(self):
        return self._residue
