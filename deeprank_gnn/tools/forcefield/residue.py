import re
from deeprank_gnn.models.forcefield.residue import (
    ResidueClassCriterium,
    ALL_AMINO_ACIDS,
)


class ResidueClassParser:
    _RESIDUE_CLASS_PATTERN = re.compile(r"([A-Z]{3,4}) *\: *name *\= *(all|[A-Z]{3})")
    _RESIDUE_ATOMS_PATTERN = re.compile(r"(present|absent)\(([A-Z0-9\, ]+)\)")

    @staticmethod
    def parse(file_):
        result = []
        for line in file_:
            match = ResidueClassParser._RESIDUE_CLASS_PATTERN.match(line)
            if not match:
                raise ValueError(f"unparsable line: '{line}'")

            class_name = match.group(1)
            amino_acid_names = ResidueClassParser._parse_amino_acids(match.group(2))

            present_atom_names = []
            absent_atom_names = []
            for match in ResidueClassParser._RESIDUE_ATOMS_PATTERN.finditer(
                line[match.end() :]
            ):
                atom_names = [name.strip() for name in match.group(2).split(",")]
                if match.group(1) == "present":
                    present_atom_names.extend(atom_names)

                elif match.group(1) == "absent":
                    absent_atom_names.extend(atom_names)

            result.append(
                ResidueClassCriterium(
                    class_name, amino_acid_names, present_atom_names, absent_atom_names
                )
            )
        return result

    @staticmethod
    def _parse_amino_acids(string):
        if string.strip() == "all":
            return ALL_AMINO_ACIDS

        return [name.strip() for name in string.split(",")]
