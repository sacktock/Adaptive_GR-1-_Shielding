from __future__ import annotations

from typing import Optional, List

from spec_repair.enums import Learning
from spec_repair.helpers.repair_nodes.repair_node import RepairNode


class CandidateRepairNode(RepairNode):
    def __init__(
            self,
            spec: list[str],
            ct_list: Optional[any],
            learning_hypothesis: Optional[list[str]],
            learning_type: Learning,
            weak_spec_history: Optional[List[list[str]]] = None,
    ):
        self.spec = spec
        self.ct_list = ct_list
        self.learning_hypothesis = learning_hypothesis
        self.learning_type = learning_type
        self.weak_spec_history = weak_spec_history if weak_spec_history else []

    def get_first_weakening_if_exists(self) -> Optional[list[str]]:
        return self.weak_spec_history[0] if self.weak_spec_history else None

    def __eq__(self, other):
        return (self.spec == other.spec and
                sorted(self.ct_list) == sorted(other.ct_list) and
                self.learning_hypothesis == other.learning_hypothesis and
                self.learning_type == other.learning_type and
                self.get_first_weakening_if_exists() == other.get_first_weakening_if_exists())

    def __hash__(self):
        hashable_learning_hypothesis = tuple(self.learning_hypothesis) if self.learning_hypothesis is not None else None
        hashable_first_weakening = tuple(self.get_first_weakening_if_exists()) if self.get_first_weakening_if_exists() \
            else None
        return hash((
            tuple(self.spec),
            tuple(sorted(self.ct_list)),
            hashable_learning_hypothesis,
            self.learning_type,
            hashable_first_weakening
        ))

    def __str__(self):
        return f"""
        Spec: {self.spec}
        CTs: {self.ct_list}
        Learning Hypothesis: {self.learning_hypothesis}
        Learning Type: {self.learning_type}
        Weak Spec History: {self.weak_spec_history}
        """
