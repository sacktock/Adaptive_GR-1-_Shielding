from enum import Enum

from spec_repair.ltl_types import GR1FormulaType


class When(Enum):
    INITIALLY = 1
    ALWAYS = 2
    EVENTUALLY = 3


class ExpType(Enum):
    ASSUMPTION = "assumption"
    GUARANTEE = "guarantee"

    def __str__(self) -> str:
        return str(self.value)


class Learning(Enum):
    ASSUMPTION_WEAKENING = "assumption weakening"
    GUARANTEE_WEAKENING = "guarantee weakening"

    def __str__(self) -> str:
        return str(self.value)

    def exp_type_str(self) -> str:
        return str(self.exp_type())

    def exp_type(self) -> ExpType:
        match self:
            case Learning.ASSUMPTION_WEAKENING:
                return ExpType.ASSUMPTION
            case Learning.GUARANTEE_WEAKENING:
                return ExpType.GUARANTEE

    def formula_type(self) -> GR1FormulaType:
        match self:
            case Learning.ASSUMPTION_WEAKENING:
                return GR1FormulaType.ASM
            case Learning.GUARANTEE_WEAKENING:
                return GR1FormulaType.GAR


class Outcome(Enum):
    NO_VIOLATION_TRACE_FOUND = 1
    REALIZABLE_SPEC_GENERATED = 2
    NO_REALIZABLE_SPEC_REACHED = 3
    UNEXPECTED_OUTCOME = 4
    # Add more if necessary


class SimEnv(Enum):
    Success = 0
    Unrealizable = 1
    Timeout = 2
    IncorrectGuarantees = 3
    NoTraceFound = 4
    Realizable = 5
    Invalid = 6

    def __str__(self) -> str:
        if self == SimEnv.Success:
            return "Success - Assumptions and Guarantees Captured."
        if self == SimEnv.Unrealizable:
            return "Unrealizable Specification Produced."
        if self == SimEnv.Timeout:
            return "Timeout."
        if self == SimEnv.IncorrectGuarantees:
            return "Environment Assumptions Captured.\n\tGuarantees Different."
        if self == SimEnv.NoTraceFound:
            return "No Violating Trace Found."
        if self == SimEnv.Realizable:
            return "Alternative Realizable Specification Produced."
        if self == SimEnv.Invalid:
            return "Invalid."

    def print(self):
        print("\nSimulated Environment Outcome:")
        print(f"\t{self}")
