from enum import Enum
from typing import Dict, Any


class PatchActionType(Enum):
    MODIFY = 1
    ADD = 2


class PatchSelection:
    def __init__(self, residue_type: str, atom_name: str):
        self.residue_type = residue_type
        self.atom_name = atom_name


class PatchAction:
    def __init__(self, type_: str, selection: PatchSelection, kwargs: Dict[str, Any]):
        self.type = type_
        self.selection = selection
        self.kwargs = kwargs

    def __contains__(self, key):
        return key in self.kwargs

    def __getitem__(self, key):
        return self.kwargs[key]
