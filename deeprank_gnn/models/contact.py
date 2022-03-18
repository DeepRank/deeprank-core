from typing import Any, Optional, Iterator

from deeprank_gnn.models.structure import Atom, Residue


class Contact:
    def __init__(self, item1: Any, item2: Any, distance: float):
        self._item1 = item1
        self._item2 = item2
        self._distance = distance

    @property
    def distance(self) -> float:
        return self._distance

    def __contains__(self, item):
        return self._item1 == item or self._item2 == item

    def __eq__(self, other):
        return type(self) == type(other) and self.distance == other.distance and \
               (self._item1 == other._item1 and self._item2 == other._item2 or \
                self._item1 == other._item2 and self._item2 == other._item1)

    def __repr__(self):
        return "{}({} - {}, {})".format(type(self), self._item1, self._item2, self._distance)

    def __iter__(self):
        return iter([self._item1, self._item2])


class ResidueContact(Contact):
    "holds a contact between two residues"

    def __init__(self, residue1: Residue, residue2: Residue, distance: float):
        Contact.__init__(self, residue1, residue2, distance)

    @property
    def residue1(self) -> Residue:
        return self._item1

    @property
    def residue2(self) -> Residue:
        return self._item2


class AtomicContact(Contact):
    "holds a contact between two atoms"

    def __init__(self, atom1: Atom, atom2: Atom,
                 distance: float,
                 coulomb_potential: Optional[float] = None,
                 vanderwaals_potential: Optional[float] = None):
        Contact.__init__(self, atom1, atom2, distance)

        self._coulomb_potential = coulomb_potential
        self._vanderwaals_potential = vanderwaals_potential

    @property
    def atom1(self) -> Atom:
        return self._item1

    @property
    def atom2(self) -> Atom:
        return self._item2

    @property
    def coulomb_potential(self) -> Optional[float]:
        return self._coulomb_potential

    @coulomb_potential.setter
    def coulomb_potential(self, potential: float):
        self._coulomb_potential = potential

    @property
    def vanderwaals_potential(self) -> Optional[float]:
        return self._vanderwaals_potential

    @vanderwaals_potential.setter
    def vanderwaals_potential(self, potential: float):
        self._vanderwaals_potential = potential
