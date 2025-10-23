import re
from abc import ABC
from typing import Callable, List, Any


# from spec_repair.helpers.counter_trace import CounterTrace


class ExceptionRule(ABC):
    pass


class AntecedentExceptionRule(ExceptionRule):
    pattern = re.compile(r"^antecedent_exception\(([^,]+,){3}[^,]+\)\s*:-\s*(not_)?holds_at\(([^,]+,){2}[^,]+\).$")


class ConsequentExceptionRule(ExceptionRule):
    pattern = re.compile(r"^consequent_exception\(([^,]+,){2}[^,]+\)\s*:-\s*(not_)?holds_at\(([^,]+,){2}[^,]+\).$")


class EventuallyConsequentRule(ExceptionRule):
    pattern = re.compile(
        r"^consequent_exception\(([^,]+,){2}[^,]+\)\s*:-\s*root_consequent_holds\(([^,]+,){4}[^,]+\).$")


class HoldsAtAtom:
    NEG_PREFIX = 1
    ATOM = 2
    pattern = re.compile(r"^(not_)?holds_at\(([^,]+),([^,]+),[^,]+\).?$")


class GR1FormulaPattern:
    TEMP_OP = 1
    FORMULA = 2
    pattern = re.compile(r"\s*(inv|alw|alwEv|G|GF)?\((.*)\);?$")

class GR1Atom:
    ATOM_TYPE = 1
    VALUE_TYPE = 2
    NAME = 3
    pattern = re.compile(r'^\s*(env|sys)\s+([a-zA-Z0-9_-]+)\s+([a-zA-Z0-9_-]+);?\s*$')

# StopHeuristicType = Callable[[List[str], List[CounterTrace]], bool]
StopHeuristicType = Callable[[List[str], List[Any]], bool]
