from enum import Enum
from typing import Set, List

import pandas as pd

class Trace:
    variables: Set[str]
    path: List[Set[str]]

    def __init__(self, file_name: str):
        raise NotImplemented

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.path):
            raise StopIteration
        value = self.path[self.index]
        self.index += 1
        return value


Trace = str
CounterStrategy = List[str]
Spec = pd.DataFrame


class GR1FormulaType(Enum):
    ASM = "assumption|asm"
    GAR = "guarantee|gar"

    def __str__(self) -> str:
        return f"{self.value}"

    def to_str(self, short_version: bool = False) -> str:
        if short_version:
            return self.value.split("|")[1]
        else:
            return self.value.split("|")[0]


    @staticmethod
    def from_str(value: str) -> "GR1FormulaType":
        if value.lower() in ["assumption", "asm"]:
            return GR1FormulaType.ASM
        elif value.lower() in ["guarantee", "gar"]:
            return GR1FormulaType.GAR
        raise ValueError(f"Unsupported value: {value}")

    def to_asp(self) -> str:
        if self == GR1FormulaType.ASM:
            return "assumption"
        elif self == GR1FormulaType.GAR:
            return "guarantee"
        else:
            raise ValueError(f"Unsupported value: {self}")


class GR1AtomType(Enum):
    SYS = "sys"
    ENV = "env"

    def __str__(self) -> str:
        return f"{self.value}"

    @staticmethod
    def from_str(value: str) -> "GR1AtomType":
        if value == "sys":
            return GR1AtomType.SYS
        elif value == "env":
            return GR1AtomType.ENV
        raise ValueError(f"Unsupported value: {value}")


class GR1TemporalType(Enum):
    INITIAL = "ini"
    INVARIANT = "G"
    JUSTICE = "GF"

    def __str__(self) -> str:
        return f"{self.value}"


class LTLFiltOperation(Enum):
    IMPLIES = "imply"
    EQUIVALENT = "equivalent-to"

    def __str__(self) -> str:
        return f"--{self.value}"

    def flag(self) -> str:
        return f"--{self.value}"
