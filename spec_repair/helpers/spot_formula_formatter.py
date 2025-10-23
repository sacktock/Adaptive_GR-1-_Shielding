from copy import deepcopy
from typing import Tuple

from py_ltl.formatter import ILTLFormatter
from py_ltl.formula import LTLFormula, AtomicProposition, Not, And, Or, Until, Next, Globally, Eventually, Implies, \
    Prev, Top, Bottom


class SpotFormulaFormatter(ILTLFormatter):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def format(self, this_formula: LTLFormula) -> str:
        # Never risk modifying the original formula
        this_formula = deepcopy(this_formula)
        spot_formula, shift = self._format(this_formula, shift_in=0)
        return spot_formula

    def _format(self, this_formula: LTLFormula, shift_in: int) -> Tuple[str, int]:
        match this_formula:
            case AtomicProposition(name=name, value=True):
                return self._apply_shift(name, shift_in), shift_in
            case AtomicProposition(name=name, value=False):
                return self._apply_shift(f"!{name}", shift_in), shift_in
            case Not(formula=formula):
                inner_formula, new_shift_out = self._format(formula, shift_in)
                return f"!({inner_formula})", new_shift_out
            case And(left=left, right=right):
                lhs, left_shift_out = self._format(left, shift_in)
                rhs, right_shift_out = self._format(right, shift_in + left_shift_out)
                lhs = self._apply_shift(lhs, right_shift_out - left_shift_out)
                return f"({lhs} & {rhs})", right_shift_out
            case Or(left=left, right=right):
                lhs, left_shift_out = self._format(left, shift_in)
                rhs, right_shift_out = self._format(right, shift_in + left_shift_out)
                lhs = self._apply_shift(lhs, right_shift_out - left_shift_out)
                return f"({lhs} | {rhs})", right_shift_out
            case Implies(left=left, right=right):
                lhs, left_shift_out = self._format(left, shift_in)
                rhs, right_shift_out = self._format(right, shift_in + left_shift_out)
                lhs = self._apply_shift(lhs, right_shift_out - left_shift_out)
                return f"({lhs} -> {rhs})", right_shift_out
            case Next(formula=formula):
                inner_formula, new_shift_out = self._format(formula, shift_in)
                return f"X({inner_formula})", new_shift_out
            case Prev(formula=formula):
                # Shift everything in the subformula by +1
                new_formula, new_shift_out = self._format(formula, max(shift_in - 1, 0))
                return new_formula, new_shift_out + 1
            case Eventually(formula=formula):
                inner_formula, new_shift_out = self._format(formula, shift_in)
                return f"F({inner_formula})", new_shift_out
            case Globally(formula=formula):
                inner_formula, new_shift_out = self._format(formula, shift_in)
                return f"G({inner_formula})", new_shift_out
            case Top():
                return "true", shift_in
            case Bottom():
                return "false", shift_in
            case _:
                raise NotImplementedError(f"Spot formatting not implemented for: {type(this_formula)}")

    def _apply_shift(self, formula_str: str, shift: int) -> str:
        for _ in range(shift):
            formula_str = f"X({formula_str})"
        return formula_str
