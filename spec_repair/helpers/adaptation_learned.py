import re
from typing import TypeVar, Optional

Self = TypeVar('T', bound='AdaptationLearned')

class Adaptation:
    """
    Represents an adaptation learned from a violation of a specification.
    It is a decoding of the output that is produced by ILASP during learning,
    and is used to structure and extract the learned adaptation.
    """

    def __init__(self, type, formula_name, disjunction_index, atom_temporal_operators):
        """
        :param type: The type of the adaptation learned (i.e. antecedent_exception, consequent_exception, etc).
        :param formula_name: The name of the invariant/justice goal that will be adapted.
        :param disjunction_index: The index of the disjunction that will be adapted within the expression.
        :param atom_temporal_operators: The atoms and their respective temporal operators used in the adaptation.
        """
        self.type: str = type
        self.formula_name: str = formula_name
        self.disjunction_index: Optional[int] = disjunction_index
        self.atom_temporal_operators = atom_temporal_operators

    def __str__(self):
        return f'{self.type}({self.formula_name},{self.disjunction_index},{self.atom_temporal_operators})'

    def __eq__(self, other):
        return (
            self.type == other.type
            and self.formula_name == other.formula_name
            and self.disjunction_index == other.disjunction_index
            and self.atom_temporal_operators == other.atom_temporal_operators
        )

    @staticmethod
    def from_str(rule: str) -> Self:
        # 1. Extract the function name at the beginning ("antecedent_exception")
        function_name = re.search(r'^(\w+)\(', rule)
        function_name = function_name.group(1) if function_name else None

        # 2. Extract the first argument of the function ("assumption2_1")
        name_expression = re.search(r'^\w+\((\w+)', rule)
        name_expression = name_expression.group(1) if name_expression else None

        # 3. Extract the number in the second argument (0 in this case)
        disjunction_index = re.search(r'^\w+\(\w+,\s*(\d+)', rule)
        disjunction_index = int(disjunction_index.group(1)) if disjunction_index else None

        # 4. Extract the first and third arguments of each "timepoint_of_op" function
        timepoint_op_args = re.findall(r'timepoint_of_op\((\w+),[^,]*,(\w+)', rule)
        # Create a dictionary to map each variable to its corresponding operator
        variable_to_operator = {var: op for op, var in timepoint_op_args}

        # Step 5: Extract both "holds_at" and "not_holds_at" calls with first and second arguments
        holds_at_args = re.findall(r'(not_)?holds_at\((\w+),\s*(\w+)', rule)

        # Construct atom string with truth values and replaced variables
        replaced_holds_at_args = [
            (
                variable_to_operator.get(arg2, arg2),
                f"{atom}={'false' if prefix == 'not_' else 'true'}"
            )
            for prefix, atom, arg2 in holds_at_args
        ]

        return Adaptation(
            type=function_name,
            formula_name=name_expression,
            disjunction_index=disjunction_index,
            atom_temporal_operators=replaced_holds_at_args
        )
