from copy import deepcopy
from typing import Optional

from spec_repair.helpers.spot_formula_formatter import SpotFormulaFormatter
from spec_repair.ltl_types import GR1FormulaType


class SpotSpecificationFormatter:
    def __init__(self, type: Optional[GR1FormulaType] = None):
        self._formula_formatter = SpotFormulaFormatter()
        self._type = type

    def format(self, spec) -> str:
        """Formats formulas from specification to SPOT format string.
        
        Args:
            spec: SpectraSpecification object containing formulas
            
        Returns:
            Formatted string of formulas joined with '&'
        """
        # Never risk modifying the original spec
        spec = deepcopy(spec)
        asm_formulas = []
        gar_formulas = []
        if self._type is not GR1FormulaType.GAR:
            for _, row in spec._formulas_df[spec._formulas_df.type == GR1FormulaType.ASM].iterrows():
                asm_formulas.append(row.formula)

        if self._type is not GR1FormulaType.ASM:
            for _, row in spec._formulas_df[spec._formulas_df.type == GR1FormulaType.GAR].iterrows():
                gar_formulas.append(row.formula)

        asms = '&'.join([f.to_str(self._formula_formatter) for f in asm_formulas])
        gars = '&'.join([f.to_str(self._formula_formatter) for f in gar_formulas])
        if asms and gars:
            formulas_str = f"({asms})->({gars})"
        elif asms:
            formulas_str = asms
        else:
            formulas_str = gars
        if not formulas_str:
            return "true"
        return formulas_str
