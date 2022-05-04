from typing import Union, List

ALL_AMINO_ACIDS = "all"  # tells the forcefield that it should match any amino acid


class ResidueClassCriterium:
    def __init__(
        self,
        class_name: str,
        amino_acid_names: Union[str, List[str]],
        present_atom_names: List[str],
        absent_atom_names: List[str],
    ):
        self.class_name = class_name

        self.amino_acid_names = amino_acid_names

        self.present_atom_names = present_atom_names
        self.absent_atom_names = absent_atom_names

    def matches(self, amino_acid_name: str, atom_names: List[str]) -> bool:

        # check the amino acid name
        if self.amino_acid_names != ALL_AMINO_ACIDS:
            if not any(

                    amino_acid_name == crit_amino_acid_name
                    for crit_amino_acid_name in self.amino_acid_names

            ):

                return False

        # check the atom names that should be absent
        if any(atom_name in self.absent_atom_names for atom_name in atom_names):

            return False

        # check the atom names that should be present
        if not all(atom_name in atom_names for atom_name in self.present_atom_names):

            return False

        # all checks passed
        return True
