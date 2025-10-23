from copy import deepcopy
from typing import List, Dict, Optional

from py_ltl.formatter import ILTLFormatter
from py_ltl.formula import LTLFormula, AtomicProposition, Not, And, Or, Until, Next, Globally, Eventually, Implies, \
    Prev, Top, Bottom

from collections import defaultdict

from spec_repair.util.formula_util import get_disjuncts_from_disjunction


class ASPExceptionFormatter(ILTLFormatter):
    def __init__(
            self,
            is_antecedent_exception: bool = False,
            is_consequent_exception: bool = False,
            is_eventually_exception: bool = False,
    ):
        self.is_antecedent_exception = is_antecedent_exception
        self.is_consequent_exception = is_consequent_exception
        self.is_eventually_exception = is_eventually_exception

    def format(self, formula: LTLFormula) -> str:
        # Never risk modifying the original formula
        formula = deepcopy(formula)
        if isinstance(formula, Top) or isinstance(formula, Bottom):
            raise ValueError("Top and Bottom are not supported in this formula")
        elif not isinstance(formula, Globally):
            return self.format_initial(formula)
        else:
            return self.format_invariant(formula.formula)

    def format_initial(self, this_formula: LTLFormula) -> str:
        match this_formula:
            case Implies(left=lhs, right=rhs):
                output = self.process_antecedent(lhs, time=0)
                output += self.process_consequent(rhs, time=0)
                return output
            case Globally(formula=formula):
                raise ValueError("Globally operator not supported in this formula")
            case _:
                output = self.antecedent_boilerplate(time=0, ops=None, antecedent_id=0)
                output += self.process_consequent(this_formula, time=0)
                return output

    def format_invariant(self, this_formula: LTLFormula) -> str:
        match this_formula:
            case Implies(left=lhs, right=rhs):
                output = self.process_antecedent(lhs, time="T")
                output += self.process_consequent(rhs, time="T")
                return output
            case _:
                output = self.antecedent_boilerplate(time="T", ops=None, antecedent_id=0)
                output += self.process_consequent(this_formula, time="T")
                return output

    def process_antecedent(self, this_formula, time):
        assert not isinstance(this_formula, Eventually)
        sections = []
        root_id = 0
        disjunction = get_disjuncts_from_disjunction(this_formula)
        for d_id, disjunct in enumerate(disjunction):
            section = ""
            ops_antecedent_roots: Dict[str, List[LTLFormula]] = reformat_conjunction_to_op_atom_conjunction(disjunct)
            section += self.antecedent_boilerplate(time=time, ops=ops_antecedent_roots.keys(), antecedent_id=d_id, start_root_id=root_id)
            for i, (_, atoms) in enumerate(ops_antecedent_roots.items()):
                section += "\n\n"
                section += self.format_boilerplate_root_antecedent_holds(atoms, root_id + i)
            root_id += len(ops_antecedent_roots)
            sections.append(section)

        output = "\n\n".join(sections)
        return output

    def process_consequent(self, this_formula, time):
        if isinstance(this_formula, Eventually):
            output = self.process_eventually_consequent(this_formula)
        else:
            output = self.process_dnf_consequent(this_formula, time)
        if self.is_consequent_exception:
            output += "\n\n"
            output += self.consequent_exception_boilerplate(time=time)
        return output

    def process_dnf_consequent(self, this_formula, time):
        assert not isinstance(this_formula, Eventually)
        output = ""
        disjunction = get_disjuncts_from_disjunction(this_formula)
        root_id = 0
        for disjunct in disjunction:
            output += "\n\n"
            ops_consequent_roots: Dict[str, List[LTLFormula]] = reformat_conjunction_to_op_atom_conjunction(disjunct)
            output += self.consequent_boilerplate(time=time, ops=ops_consequent_roots.keys(), start_root_id=root_id)
            if time != 0 and self.is_eventually_exception:
                output += "\n\n"
                ops = ["eventually"] * len(ops_consequent_roots)
                output += self.consequent_boilerplate(time=time, ops=ops, start_root_id=root_id)
            for i, (_, atoms) in enumerate(ops_consequent_roots.items()):
                output += "\n\n"
                output += self.format_boilerplate_root_consequent_holds(atoms, root_id + i)
            root_id += len(ops_consequent_roots)
        return output

    def process_eventually_consequent(self, this_formula):
        assert isinstance(this_formula, Eventually)
        output = ""
        disjunction = get_disjuncts_from_disjunction(this_formula.formula)
        root_id = 0
        for disjunct in disjunction:
            output += "\n\n"
            ops_consequent_roots: Dict[str, List[LTLFormula]] = reformat_conjunction_to_op_atom_conjunction(disjunct)
            ops = ["eventually"] * len(ops_consequent_roots)
            output += self.consequent_boilerplate(time="T", ops=ops, start_root_id=root_id)
            output = output.replace(",\n\tev_temp_op({name})","")
            for i, (_, atoms) in enumerate(ops_consequent_roots.items()):
                output += "\n\n"
                output += self.format_boilerplate_root_consequent_holds(atoms, root_id + i)
            root_id += len(ops_consequent_roots)
        return output

    def format_exp(self, this_formula: LTLFormula) -> str:
        assert not isinstance(this_formula, Globally)
        match this_formula:
            case AtomicProposition(name=name, value=value):
                if value is True:
                    return f"holds_at({name},T2,S)"
                elif value is False:
                    return f"not_holds_at({name},T2,S)"
                else:
                    raise ValueError(f"Unsupported value for atomic proposition: {value}")
            case Not(formula=formula):
                if isinstance(formula, AtomicProposition):
                    formula.value = not formula.value
                    return self.format_exp(formula)
                else:
                    raise ValueError("Not operator not supported for this formula")
            case And(left=lhs, right=rhs):
                ops_atoms = reformat_conjunction_to_op_atom_conjunction(this_formula)
                output = self.format_boilerplate_holds(ops_atoms)
                for i, (_, atoms) in enumerate(ops_atoms.items()):
                    output += "\n\n"
                    output += self.format_boilerplate_root_consequent_holds(atoms, i)
                return output
            case Or(left=lhs, right=rhs):
                return f"({self.format(lhs)}|{self.format(rhs)})"
            case Implies(left=lhs, right=rhs):
                antecedent = complete_implication_part(self.format(lhs), "antecedent")
                consequent = complete_implication_part(self.format(rhs), "consequent")

                return f"{antecedent}\n\n{consequent}"
            case Next(formula=formula):
                ops_atoms = reformat_conjunction_to_op_atom_conjunction(this_formula)
                output = self.format_boilerplate_holds(ops_atoms)
                for i, (_, atoms) in enumerate(ops_atoms.items()):
                    output += "\n\n"
                    output += self.format_boilerplate_root_consequent_holds(atoms, i)
                return output
            case Prev(formula=formula):
                ops_atoms = reformat_conjunction_to_op_atom_conjunction(this_formula)
                output = self.format_boilerplate_holds(ops_atoms)
                for i, (_, atoms) in enumerate(ops_atoms.items()):
                    output += "\n\n"
                    output += self.format_boilerplate_root_consequent_holds(atoms, i)
                return output
            case Eventually(formula=formula):
                ops_atoms = reformat_conjunction_to_op_atom_conjunction(this_formula)
                output = self.format_boilerplate_holds(ops_atoms)
                for i, (_, atoms) in enumerate(ops_atoms.items()):
                    output += "\n\n"
                    output += self.format_boilerplate_root_consequent_holds(atoms, i)
                return output.replace("{implication_type}", "consequent")
            case Globally(formula=formula):
                if isinstance(formula, Eventually):
                    return f"G{self.format(formula)}"
                return f"G({self.format(formula)})"
            case Top():
                return ""
            case Bottom():
                return None
            case _:
                raise NotImplementedError(f"Formatter not implemented for: {type(this_formula)}")

    @staticmethod
    def format_boilerplate_holds(ops_atoms):
        output = f"{{implication_type}}_holds({{name}},T,S):-\n"
        output += "\ttrace(S),\n"
        output += "\ttimepoint(T,S)"
        for i, (op, atoms) in enumerate(ops_atoms.items()):
            output += f",\n\troot_{{implication_type}}_holds({op},{{name}},{i},T,S)"
        output += "."
        return output

    def format_boilerplate_root_antecedent_holds(self, atoms, i):
        output = f"root_antecedent_holds(OP,{{name}},{i},T1,S):-\n"
        output += "\ttrace(S),\n"
        output += "\ttimepoint(T1,S),\n"
        output += "\tnot weak_timepoint(T1,S),\n"
        output += "\ttimepoint(T2,S),\n"
        output += "\ttemporal_operator(OP),\n"
        output += "\ttimepoint_of_op(OP,T1,T2,S)"
        for atom in atoms:
            output += f",\n\t{self.format_exp(atom)}"
        output += "."
        return output

    def format_boilerplate_root_consequent_holds(self, atoms, i):
        output = f"root_consequent_holds(OP,{{name}},{i},T1,S):-\n"
        output += "\ttrace(S),\n"
        output += "\ttimepoint(T1,S),\n"
        output += "\tnot weak_timepoint(T1,S),\n"
        output += "\ttimepoint(T2,S),\n"
        output += "\ttemporal_operator(OP),\n"
        output += "\ttimepoint_of_op(OP,T1,T2,S)"
        for atom in atoms:
            output += f",\n\t{self.format_exp(atom)}"
        output += "."
        return output

    def antecedent_boilerplate(self, time, ops, antecedent_id, start_root_id=0):
        output = f"""\
antecedent_holds({{name}},{time},S):-
\ttrace(S),
\ttimepoint({time},S)\
"""
        if ops is not None:
            for i, op in enumerate(ops):
                output += f",\n\troot_antecedent_holds({op},{{name}},{start_root_id + i},{time},S)"
        if self.is_antecedent_exception:
            output += f",\n\tnot antecedent_exception({{name}},{antecedent_id},{time},S)"
        return f"{output}."

    def consequent_boilerplate(self, time, ops, start_root_id=0):
        output = f"""\
consequent_holds({{name}},{time},S):-
\ttrace(S),
\ttimepoint({time},S)\
"""
        if ops is not None:
            for i, op in enumerate(ops):
                output += f",\n\troot_consequent_holds({op},{{name}},{start_root_id + i},{time},S)"
        if time != 0 and self.is_eventually_exception:
            if "eventually" not in ops:
                output += f",\n\tnot ev_temp_op({{name}})"
            else:
                output += f",\n\tev_temp_op({{name}})"
        return f"{output}."

    def consequent_exception_boilerplate(self, time):
        return f"""\
consequent_holds({{name}},{time},S):-
\ttrace(S),
\ttimepoint({time},S),
\tconsequent_exception({{name}},{time},S).\
"""

def reformat_conjunction_to_op_atom_conjunction(this_formula) -> Dict[str, List[LTLFormula]]:
    match this_formula:
        case AtomicProposition(name=name, value=value):
            return {"current": [this_formula]}
        case Not(formula=formula):
            if isinstance(formula, AtomicProposition):
                return {"current": [this_formula]}
            else:
                raise ValueError("Not operator not supported for this formula")
        case And(left=lhs, right=rhs):
            lhs_format = reformat_conjunction_to_op_atom_conjunction(lhs)
            rhs_format = reformat_conjunction_to_op_atom_conjunction(rhs)
            return merge_dicts(lhs_format, rhs_format)
        case Or(left=lhs, right=rhs):
            raise ValueError("Or operator not supported for this operation")
        case Implies(left=lhs, right=rhs):
            raise ValueError("Implies operator not supported for this operation")
        case Next(formula=formula):
            if isinstance(formula, And):
                inner_format = reformat_conjunction_to_op_atom_conjunction(formula)
                assert inner_format.keys() == {"current"}
                return {"next": inner_format["current"]}
            assert isinstance(formula, AtomicProposition) or isinstance(formula, Not) or isinstance(formula, Top) or isinstance(formula, Bottom)
            return {"next": [formula]}
        case Prev(formula=formula):
            if isinstance(formula, And):
                inner_format = reformat_conjunction_to_op_atom_conjunction(formula)
                assert inner_format.keys() == {"current"}
                return {"prev": inner_format["current"]}
            assert isinstance(formula, AtomicProposition) or isinstance(formula, Not) or isinstance(formula, Top) or isinstance(formula, Bottom)
            return {"prev": [formula]}
        case Eventually(formula=formula):
            if isinstance(formula, And):
                inner_format = reformat_conjunction_to_op_atom_conjunction(formula)
                assert inner_format.keys() == {"current"}
                return {"eventually": inner_format["current"]}
            assert isinstance(formula, AtomicProposition) or isinstance(formula, Not) or isinstance(formula, Top) or isinstance(formula, Bottom)
            return {"eventually": [formula]}
        case Globally(formula=formula):
            raise ValueError("Implies operator not supported for this operation")
        case Top():
            return {}
        case Bottom():
            raise ValueError("Bottom operator not supported for this operation")
            # return {"current": [this_formula]}
        case _:
            raise NotImplementedError(f"Reformatter not implemented for: {type(this_formula)}")


def complete_implication_part(formatted_string, implication_type: str):
    assert implication_type in ["antecedent", "consequent"]



def merge_dicts(*dicts):
    result = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            result[key].extend(value)
    return dict(result)