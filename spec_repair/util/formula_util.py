from functools import reduce
from typing import List, Optional

from py_ltl.formula import LTLFormula, AtomicProposition, Not, And, Or, Until, Next, Globally, Eventually, Implies, Prev, Top, \
    Bottom


def get_temp_op(this_formula: LTLFormula) -> str:
    match this_formula:
        case Prev(formula=formula):
            return "prev"
        case Next(formula=formula):
            return "next"
        case AtomicProposition(name=name, value=value):
            return "current"
        case Eventually(formula=formula):
            return "eventually"
        case Not(formula=formula):
            return get_temp_op(formula)
        case Or(left=left, right=right):
            raise ValueError("Or operator not supported in this formula")
        case And(left=left, right=right):
            raise ValueError("And operator not supported in this formula")
        case Implies(left=left, right=right):
            raise ValueError("Implies operator not supported in this formula")
        case Globally(formula=formula):
            raise ValueError("Globally operator not supported in this formula")
        case Until(left=left, right=right):
            raise ValueError("Until operator not supported anywhere")
        case Top():
            return "current"
        case Bottom():
            return "current"
        case _:
            raise ValueError(f"Unsupported formula type {type(this_formula)}")

def get_disjuncts_from_disjunction(disjunction: Optional[LTLFormula]) -> List[LTLFormula]:
    if not disjunction:
        return []
    disjuncts = []
    while isinstance(disjunction, Or):
        disjuncts.append(disjunction.right)
        disjunction = disjunction.left
    disjuncts.append(disjunction)
    disjuncts.reverse()
    return disjuncts

def get_conjuncts_from_conjunction(conjunction: Optional[LTLFormula]) -> List[LTLFormula]:
    if not conjunction:
        return []
    conjuncts = []
    while isinstance(conjunction, And):
        conjuncts.append(conjunction.right)
        conjunction = conjunction.left
    conjuncts.append(conjunction)
    conjuncts.reverse()
    return conjuncts

def disjoin_all(formulas: list[LTLFormula]) -> LTLFormula:
    if not formulas:
        raise ValueError("Cannot disjoin an empty list of formulas")
    return reduce(lambda a, b: Or(a, b), formulas)


def skip_first_temp_op(this_formula: LTLFormula) -> LTLFormula:
    match this_formula:
        case Prev(formula=formula):
            return formula
        case Next(formula=formula):
            return formula
        case Eventually(formula=formula):
            return formula
        case Globally(formula=formula):
            return formula
        case AtomicProposition(name=name, value=value):
            return this_formula
        case Not(_):
            return this_formula
        case Or(_, _):
            return this_formula
        case And(_, _):
            return this_formula
        case Implies(_, _):
            return this_formula
        case Top():
            return this_formula
        case Bottom():
            return this_formula
        case Until(_, _):
            raise ValueError("Until operator not supported anywhere")
        case _:
            raise ValueError(f"Unsupported formula type {type(this_formula)}")

# see journal paper
def is_ilasp_compatible_dnf_structure(disjunction_of_conjunctions) -> bool:
    is_response = False
    if isinstance(disjunction_of_conjunctions, Eventually):
        disjunction_of_conjunctions = disjunction_of_conjunctions.formula
        is_response = True
    conjunctions = get_disjuncts_from_disjunction(disjunction_of_conjunctions)
    for conjunction in conjunctions:
        if isinstance(conjunction, Or):
            return False
        conjuncts = get_conjuncts_from_conjunction(conjunction)
        for conjunct in conjuncts:
            if isinstance(conjunct, Or) or isinstance(conjunct, And) or isinstance(conjunct, Implies):
                return False
            if isinstance(conjunct, Prev) or isinstance(conjunct, Next):
                if is_response:
                    return False
                conjunct = conjunct.formula
            if isinstance(conjunct, Not):
                conjunct = conjunct.formula
            if not isinstance(conjunct, AtomicProposition):
                return False
    return True



