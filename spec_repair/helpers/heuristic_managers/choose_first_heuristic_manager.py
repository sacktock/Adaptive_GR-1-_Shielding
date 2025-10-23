from typing import List

from spec_repair.helpers.counter_trace import CounterTrace
from spec_repair.helpers.heuristic_managers.iheuristic_manager import IHeuristicManager
from spec_repair.heuristics import first_choice, choose_one_with_heuristic


class ChooseFirstHeuristicManager(IHeuristicManager):
    def select_counter_traces(self, cts: List[CounterTrace]) -> List[CounterTrace]:
        return [choose_one_with_heuristic(cts, first_choice)]

    def select_alternative_learning_tasks(self, ctss: List[List[CounterTrace]]) -> List[List[CounterTrace]]:
        return [choose_one_with_heuristic(ctss, first_choice)]

    def select_possible_learning_adaptations(self, adaptations: List[List[str]]) -> List[List[str]]:
        return [choose_one_with_heuristic(adaptations, first_choice)]
