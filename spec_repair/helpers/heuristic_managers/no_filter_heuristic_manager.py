from typing import List

from spec_repair.helpers.counter_trace import CounterTrace
from spec_repair.helpers.heuristic_managers.iheuristic_manager import IHeuristicManager


class NoFilterHeuristicManager(IHeuristicManager):
    def __init__(self):
        super().__init__()

    def select_counter_traces(self, cts: List[CounterTrace]) -> List[CounterTrace]:
        return cts

    def select_alternative_learning_tasks(self, ctss: List[List[CounterTrace]]) -> List[List[CounterTrace]]:
        return ctss

    def select_possible_learning_adaptations(self, adaptations: List[List[str]]) -> List[List[str]]:
        return adaptations
