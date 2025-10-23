from spec_repair.ltl_types import GR1AtomType
from spec_repair.special_types import GR1Atom


class SpectraAtom:
    def __init__(self, name: str, value_type: str, atom_type: GR1AtomType):
        self.name = name
        self.value_type = value_type
        self.atom_type = atom_type

    @staticmethod
    def from_str(atom_str: str):
        atom_definition = GR1Atom.pattern.match(atom_str)
        if atom_definition:
            name = atom_definition.group(GR1Atom.NAME)
            value_type = atom_definition.group(GR1Atom.VALUE_TYPE)
            atom_type = atom_definition.group(GR1Atom.ATOM_TYPE)
            return SpectraAtom(name, value_type, GR1AtomType.from_str(atom_type))
        return None

    def __str__(self):
        return f"{self.atom_type} {self.value_type} {self.name}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (self.name == other.name and
                self.value_type == other.value_type and
                self.atom_type == other.atom_type)

    def __hash__(self):
        return hash((self.name, self.value_type, self.atom_type))

    def __lt__(self, other):
        return self.name < other.name
