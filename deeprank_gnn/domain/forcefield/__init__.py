import os
import logging
import numpy
from ...models.structure import Atom, Residue
from ...models.forcefield.patch import PatchActionType
from ...tools.forcefield.top import TopParser
from ...tools.forcefield.patch import PatchParser
from ...tools.forcefield.residue import ResidueClassParser
from ...tools.forcefield.param import ParamParser
from ...models.error import UnknownAtomError

_log = logging.getLogger(__name__)


_forcefield_directory_path = os.path.dirname(os.path.abspath(__file__))


VANDERWAALS_DISTANCE_OFF = 20.0
VANDERWAALS_DISTANCE_ON = 1.0

SQUARED_VANDERWAALS_DISTANCE_OFF = numpy.square(VANDERWAALS_DISTANCE_OFF)
SQUARED_VANDERWAALS_DISTANCE_ON = numpy.square(VANDERWAALS_DISTANCE_ON)

EPSILON0 = 1.0
COULOMB_CONSTANT = 332.0636
MAX_COVALENT_DISTANCE = 3.0

class AtomicForcefield:
    def __init__(self):
        top_path = os.path.join(
            _forcefield_directory_path,
            "protein-allhdg5-5_new.top")
        with open(top_path, 'rt', encoding = 'utf-8') as f:
            self._top_rows = {(row.residue_name, row.atom_name): row for row in TopParser.parse(f)}

        patch_path = os.path.join(_forcefield_directory_path, "patch.top")
        with open(patch_path, 'rt', encoding = 'utf-8') as f:
            self._patch_actions = PatchParser.parse(f)

        residue_class_path = os.path.join(
            _forcefield_directory_path, "residue-classes")
        with open(residue_class_path, 'rt', encoding = 'utf-8') as f:
            self._residue_class_criteria = ResidueClassParser.parse(f)

        param_path = os.path.join(
            _forcefield_directory_path,
            "protein-allhdg5-4_new.param")
        with open(param_path, 'rt', encoding = 'utf-8') as f:
            self._vanderwaals_parameters = ParamParser.parse(f)

    def _find_matching_residue_class(self, residue: Residue):
        for criterium in self._residue_class_criteria:
            if criterium.matches(
                residue.amino_acid.three_letter_code, [
                    atom.name for atom in residue.atoms]):
                return criterium.class_name

        return None

    def get_vanderwaals_parameters(self, atom: Atom):
        type_ = self._get_type(atom)

        return self._vanderwaals_parameters[type_]

    def _get_type(self, atom: Atom):
        atom_name = atom.name

        if atom.residue.amino_acid is None:
            raise UnknownAtomError(f"no amino acid for {atom}")

        residue_name = atom.residue.amino_acid.three_letter_code

        type_ = None

        # check top
        top_key = (residue_name, atom_name)
        if top_key in self._top_rows:
            type_ = self._top_rows[top_key]["type"]

        # check patch, which overrides top
        residue_class = self._find_matching_residue_class(atom.residue)
        if residue_class is not None:
            for action in self._patch_actions:
                if action.type in [PatchActionType.MODIFY, PatchActionType.ADD] and \
                        residue_class == action.selection.residue_type and "TYPE" in action:

                    type_ = action["TYPE"]

        if type_ is None:
            raise UnknownAtomError(
                f"not mentioned in top or patch: {top_key}")

        return type_

    def get_charge(self, atom: Atom):
        """
            Args:
                atom(Atom): the atom to get the charge for
            Returns(float): the charge of the given atom
        """

        atom_name = atom.name
        amino_acid_code = atom.residue.amino_acid.three_letter_code

        charge = None

        # check top
        top_key = (amino_acid_code, atom_name)
        if top_key in self._top_rows:
            charge = float(self._top_rows[top_key]["charge"])

        # check patch, which overrides top
        residue_class = self._find_matching_residue_class(atom.residue)
        if residue_class is not None:
            for action in self._patch_actions:
                if action.type in [
                        PatchActionType.MODIFY,
                        PatchActionType.ADD] and residue_class == action.selection.residue_type:

                    charge = float(action["CHARGE"])

        if charge is None:
            raise UnknownAtomError(
                f"not mentioned in top or patch: {top_key}")

        return charge


atomic_forcefield = AtomicForcefield()
