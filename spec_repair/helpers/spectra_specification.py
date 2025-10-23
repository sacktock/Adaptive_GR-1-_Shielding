import copy
import os
import re
import subprocess
from copy import deepcopy
from typing import TypedDict, Optional, TypeVar, List, Set, Any, Callable

import pandas as pd
import spot

from spec_repair.components.interfaces.ispecification import ISpecification
from spec_repair.helpers.adaptation_learned import Adaptation
from spec_repair.helpers.asp_exception_formatter import ASPExceptionFormatter
from spec_repair.helpers.heuristic_managers.iheuristic_manager import IHeuristicManager
from spec_repair.helpers.heuristic_managers.no_filter_heuristic_manager import NoFilterHeuristicManager
from spec_repair.helpers.spectra_atom import SpectraAtom
from spec_repair.helpers.gr1_formula import GR1Formula
from spec_repair.helpers.spectra_formula_formatter import SpectraFormulaFormatter
from spec_repair.helpers.spectra_formula_parser import SpectraFormulaParser
from spec_repair.helpers.spot_specification_formatter import SpotSpecificationFormatter
from spec_repair.ltl_types import GR1FormulaType, GR1TemporalType
from spec_repair.util.file_util import read_file_lines, validate_spectra_file
from spec_repair.util.formula_util import get_disjuncts_from_disjunction
from spec_repair.util.spec_util import format_spec


class FormulaDataPoint(TypedDict):
    name: str
    type: GR1FormulaType
    when: GR1TemporalType
    formula: "GR1Formula"  # Use the class name as a string for forward declaration


Self = TypeVar('T', bound='SpectraSpecification')


class SpectraSpecification(ISpecification):
    _response_pattern = """\
    pattern pRespondsToS(s, p) {
      var { S0, S1} state;

      // initial assignments: initial state
      ini state=S0;

      // safety this and next state
      alw ((state=S0 & ((!s) | (s & p)) & next(state=S0)) |
      (state=S0 & (s & !p) & next(state=S1)) |
      (state=S1 & (p) & next(state=S0)) |
      (state=S1 & (!p) & next(state=S1)));

      // equivalence of satisfaction
      alwEv (state=S0);
    }"""

    def __init__(self, spec_txt: str):
        spec_txt = copy.deepcopy(spec_txt)
        self._formulas_df: pd.DataFrame = None
        self._module_name: str
        self._atoms: Set[SpectraAtom] = set()
        self._parser = SpectraFormulaParser()
        self._formater = SpectraFormulaFormatter()
        self._asp_formatter = ASPExceptionFormatter()
        formula_list = []
        spec_lines = spec_txt.splitlines()
        try:
            for i, line in enumerate(spec_lines):
                if line.find("module") >= 0:
                    self._module_name = line.split()[1]
                elif line.find("--") >= 0:
                    name: str = re.search(r'--\s*(\S+)', line).group(1)
                    type_txt: str = re.search(r'\s*(asm|assumption|gar|guarantee)\s*--', line).group(1)
                    type: GR1FormulaType = GR1FormulaType.from_str(type_txt)
                    formula_txt: str = re.sub('\s*', '', spec_lines[i + 1])
                    formula: GR1Formula = GR1Formula.from_str(formula_txt, self._parser)
                    when: GR1TemporalType = formula.temp_type
                    formula_list.append([name, type, when, formula])
                else:
                    atom: Optional[SpectraAtom] = SpectraAtom.from_str(line)
                    if atom:
                        self._atoms.add(atom)

        except AttributeError as e:
            raise e

        self._formulas_df = pd.DataFrame(formula_list, columns=["name", "type", "when", "formula"])

    def integrate_multiple(self, adaptations: List[Adaptation]):
        for adaptation in adaptations:
            self.integrate(adaptation)
        return self

    def integrate(self, adaptation: Adaptation):
        formula = self.get_formula(adaptation.formula_name)
        print("Rule:")
        print(f'\t{formula.to_str(self._formater)}')
        print("Hypothesis:")
        print(
            f'\t{adaptation.type}({adaptation.formula_name},{adaptation.disjunction_index},{adaptation.atom_temporal_operators})')
        formula.integrate(adaptation)
        print("New Rule:")
        print(f'\t{formula.to_str(self._formater)}')
        self.replace_formula(adaptation.formula_name, formula)

    def replace_formula(self, formula_name, formula):
        self._formulas_df.loc[self._formulas_df["name"] == formula_name, "formula"] = formula

    def get_formula(self, name: str):
        # Get formula by name
        formula: GR1Formula = \
            self._formulas_df.loc[self._formulas_df["name"] == name, "formula"].iloc[0]
        return formula

# TODO: make it count the amount of conjunctions with different temporal operators (max=3/disjunct)
    def get_max_disjuncts_in_antecedent(self) -> int:
        """
        Get the maximum number of conjunctions in the antecedent of any formula.
        """
        max_disjuncts = 0
        for _, row in self._formulas_df.iterrows():
            if row['type'] == GR1FormulaType.ASM:
                formula = row.formula
                antecedent = formula.antecedent
                disjuncts = get_disjuncts_from_disjunction(antecedent)
                max_disjuncts = max(max_disjuncts, len(disjuncts))
        return max_disjuncts

    @staticmethod
    def from_file(spec_file: str) -> Self:
        validate_spectra_file(spec_file)
        spec_txt: str = "".join(format_spec(read_file_lines(spec_file)))
        return SpectraSpecification(spec_txt)

    @staticmethod
    def from_str(spec_text: str) -> Self:
        spec_txt: str = "".join(format_spec(spec_text.splitlines(keepends=True)))
        return SpectraSpecification(spec_txt)

    def get_atoms(self):
        return deepcopy(self._atoms)

    def to_formatted_string(
            self,
            formatter
    ) -> str:
        return formatter.format(self)

    def to_asp(
            self,
            learning_names: Optional[List[str]] = None,
            for_clingo: bool = False,
            hm: IHeuristicManager = NoFilterHeuristicManager()
    ) -> str:
        if learning_names is None:
            learning_names = []
        formulas_str = ""
        for _, row in self._formulas_df.iterrows():
            formulas_str += self._formula_to_asp_str(row, learning_names, for_clingo, hm)
            formulas_str += "\n\n"
        return formulas_str

    def _formula_to_asp_str(self, row, learning_names, for_clingo, hm: IHeuristicManager):
        if row.when == GR1TemporalType.JUSTICE and row['name'] not in learning_names and not for_clingo:
            return ""
        formula: GR1Formula = row.formula
        expression_string = f"%{row.type.to_asp()} -- {row['name']}\n"
        expression_string += f"%\t{formula.to_str(self._formater)}\n\n"
        expression_string += f"{row.type.to_asp()}({row['name']}).\n\n"
        is_exception = (row['name'] in learning_names) and not for_clingo
        ant_exception = is_exception and hm.is_enabled("ANTECEDENT_WEAKENING")
        gar_exception = is_exception and hm.is_enabled("CONSEQUENT_WEAKENING")
        ev_exception = is_exception and hm.is_enabled("INVARIANT_TO_RESPONSE_WEAKENING")
        self._asp_formatter.is_antecedent_exception = ant_exception
        self._asp_formatter.is_consequent_exception = gar_exception
        self._asp_formatter.is_eventually_exception = ev_exception
        expression_string += row.formula.to_str(self._asp_formatter).replace("{name}", row['name'])
        return expression_string

    def filter(self, func: Callable[[pd.DataFrame], bool]) -> pd.DataFrame:
        return self._formulas_df.loc[func(self._formulas_df)]

    def extract_sub_specification(self, func: Callable[[pd.DataFrame], bool]) -> Any:
        sub_spec = deepcopy(self)
        sub_spec._formulas_df = deepcopy(self.filter(func))
        return sub_spec

    def to_str(self, is_to_compile: bool = False) -> str:
        """
        Convert the specification to a string representation.
        """
        spec_str = f"module {self._module_name}\n\n"
        for atom in sorted(self._atoms):
            spec_str += f"{atom.atom_type} {atom.value_type} {atom.name};\n"
        spec_str += "\n\n"

        self._formater.is_response_pattern = is_to_compile

        for _, row in self._formulas_df.iterrows():
            spec_str += f"{row['type'].to_str()} -- {row['name']}\n"
            spec_str += f"\t{row['formula'].to_str(self._formater)};\n\n"

        if is_to_compile and "pRespondsToS" in spec_str:
            spec_str += self._response_pattern
        self._formater.is_response_pattern = False
        return spec_str

    def __deepcopy__(self, memo):
        new_spec = SpectraSpecification("")
        new_spec._module_name = self._module_name
        new_spec._formulas_df = self._formulas_df.copy(deep=True)
        for col in new_spec._formulas_df.columns:
            if new_spec._formulas_df[col].dtype == 'O':  # Object dtype means it might contain class instances
                new_spec._formulas_df[col] = new_spec._formulas_df[col].apply(lambda x: deepcopy(x, memo))
        new_spec._atoms = deepcopy(self._atoms, memo)
        return new_spec

    def __hash__(self) -> int:
        """
        Generate a hash for the specification based on its module name and formulas.
        """
        return hash((self._module_name, tuple(self._formulas_df.itertuples(index=False, name=None))))

    def __eq__(self, other) -> bool:
        return (self.equivalent_to(other, GR1FormulaType.ASM)
                and
                self.equivalent_to(other, GR1FormulaType.GAR))

    def __ne__(self, other) -> bool:
        # Define the not equal comparison
        return not self.__eq__(other)

    def __le__(self, other):
        """
        Check if this specification is equivalent to or implies the other specification.
        """
        return self.equivalent_to(other) or self.implies(other)

    def __lt__(self, other):
        """
        Check if this specification implies the other specification.
        """
        return self.implies(other)

    def __ge__(self, other):
        """
        Check if this specification is equivalent to or is implied by the other specification.
        """
        return self.equivalent_to(other) or self.implied_by(other)

    def __gt__(self, other):
        """
        Check if this specification is implied by the other specification.
        """
        return self.implied_by(other)

    def is_trivial_true(self, formula_type: Optional[GR1FormulaType]=None) -> bool:
        return self.is_equivalent_to_spot("G(true)", formula_type)

    def is_trivial_false(self, formula_type: Optional[GR1FormulaType]=None) -> bool:
        return self.is_equivalent_to_spot("G(false)", formula_type)

    def equivalent_to(self, other, formula_type: Optional[GR1FormulaType] = None) -> bool:
        f1 = spot.formula(self.to_formatted_string(SpotSpecificationFormatter(formula_type)))
        f2 = spot.formula(other.to_formatted_string(SpotSpecificationFormatter(formula_type)))
        return spot.are_equivalent(f1, f2)

    def implies(self, other, formula_type: Optional[GR1FormulaType] = None) -> bool:
        f1 = self.to_formatted_string(SpotSpecificationFormatter(formula_type))
        f2 = other.to_formatted_string(SpotSpecificationFormatter(formula_type))
        return does_left_imply_right(f1, f2)

    def implied_by(self, other, formula_type: Optional[GR1FormulaType] = None) -> bool:
        return other.implies(self, formula_type)

    def is_equivalent_to_spot(self, formula: str, formula_type: Optional[GR1FormulaType]):
        f1 = spot.formula(self.to_formatted_string(SpotSpecificationFormatter(formula_type)))
        f2 = spot.formula(formula)
        return spot.are_equivalent(f1, f2)


def does_left_imply_right(left_exp: str, right_exp: str) -> bool:
    # TODO: introduce an assertion against ltl_ops which do not exist yet
    linux_cmd = ["ltlfilt", "-c", "-f", f"{left_exp}", "--imply", f"{right_exp}"]
    p = subprocess.Popen(linux_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    output: str = p.communicate()[0].decode('utf-8')
    reg = re.search(r"([01])\n", output)
    if not reg:
        raise Exception(
            f"The output of ltlfilt is unexpected, ergo error occurred during the comparison of:\n{left_exp}\nand\n{right_exp}",
        )
    result = reg.group(1)
    return result == "1"

