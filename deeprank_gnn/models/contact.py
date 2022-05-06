from abc import ABC

import numpy

from .pair import Pair
from .structure import Residue, Atom

class Contact(Pair, ABC):
    pass


class ResidueContact(Contact):
    "A contact between two residues from a structure"

    def __init__(self, residue1: Residue, residue2: Residue):

        self._residue1 = residue1
        self._residue2 = residue2

        super().__init__(residue1, residue2)

    @property
    def residue1(self) -> Residue:
        return self.item1

    @property
    def residue2(self) -> Residue:
        return self.item2


class AtomicContact(Contact):
    "A contact between two atoms from a structure"

    def __init__(
        self,
        atom1: Atom,
        atom2: Atom
    ):

        self._atom1 = atom1
        self._atom2 = atom2

        super().__init__(atom1, atom2)

    @property
    def atom1(self) -> Atom:
        return self.item1

    @property
    def atom2(self) -> Atom:
        return self.item2
