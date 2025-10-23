from typing import List

from spec_repair.helpers.heuristic_managers.no_filter_heuristic_manager import NoFilterHeuristicManager


class NoEventuallyHypothesisHeuristicManager(NoFilterHeuristicManager):

    def __init__(self):
        super().__init__()
        self._heuristics["INVARIANT_TO_RESPONSE_WEAKENING"] = False

    def select_possible_learning_adaptations(self, adaptations: List[List[str]]) -> List[List[str]]:
        adaptations = super().select_possible_learning_adaptations(adaptations)
        return [h for h in adaptations if 'ev_temp_op' not in "".join(h)]
