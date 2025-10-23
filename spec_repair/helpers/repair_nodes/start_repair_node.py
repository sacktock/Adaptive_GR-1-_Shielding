from __future__ import annotations

from copy import deepcopy
from typing import Optional

from spec_repair.enums import Learning
from spec_repair.helpers.repair_nodes.candidate_repair_node import CandidateRepairNode
from spec_repair.helpers.repair_nodes.repair_node import RepairNode


class StartRepairNode(RepairNode):
    def __init__(
            self,
            spec: list[str],
            ct_list: Optional[any],
            learning_type: Learning
    ):
        self.spec = spec
        self.ct_list = ct_list
        self.learning_type = learning_type
        self.weak_spec_history = []

    def __eq__(self, other) -> bool:
        return (self.spec == other.spec and
                self.ct_list == other.ct_list and
                self.learning_type == other.learning_type)

    def __hash__(self) -> int:
        hashable_first_weakening = tuple(
            self.get_first_weakening_if_exists()) if self.get_first_weakening_if_exists() else None
        return hash((
            tuple(self.spec),
            tuple(sorted(self.ct_list)),
            self.learning_type,
            hashable_first_weakening
        ))

    def get_first_weakening_if_exists(self) -> Optional[list[str]]:
        return self.weak_spec_history[0] if self.weak_spec_history else None

    def __str__(self):
        return f"""
        Spec: {self.spec}
        CTs: {self.ct_list}
        Learning Type: {self.learning_type}
        Weak Spec History: {self.weak_spec_history}
        """

    def get_candidate_repair_node(self, learning_hypothesis: list[str]) -> CandidateRepairNode:
        return CandidateRepairNode(
            deepcopy(self.spec),
            deepcopy(self.ct_list),
            learning_hypothesis,
            deepcopy(self.learning_type),
            deepcopy(self.weak_spec_history)
        )
