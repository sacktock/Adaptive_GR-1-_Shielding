from copy import deepcopy
from typing import List, Tuple, Callable, Dict

from spec_repair.components.interfaces.imitigator import IMitigator
from spec_repair.components.interfaces.ispecification import ISpecification
from spec_repair.enums import Learning
from spec_repair.helpers.counter_trace import CounterTrace
from spec_repair.helpers.heuristic_managers.no_filter_heuristic_manager import NoFilterHeuristicManager
from spec_repair.helpers.spectra_specification import SpectraSpecification


class LearningTypeSpecMitigator(IMitigator):
    """
    During learning, if no solution given, this mitigator will select one of the
    strategies it has been instantiated with based on the learning_type of the learning task.
    e.g. learning_type==Learning.ASSUMPTION_WEAKENING -> move_to_guarantee_weakening
    """
    def __init__(self, learning_strategies: Dict[Learning, Callable[[ISpecification, list[str], list[CounterTrace], list[ISpecification], int, float], List[Tuple[ISpecification, Tuple[list[str], list[CounterTrace], Learning, list[ISpecification], int, float]]]]]):
        self._hm = NoFilterHeuristicManager()
        self._mitigation_strategies = learning_strategies

    def prepare_alternative_learning_tasks(
            self,
            spec: SpectraSpecification,
            data: Tuple[list[str], list[CounterTrace], Learning, list[SpectraSpecification], int, float]
    ) -> List[Tuple[ISpecification, Tuple[list[str], list[CounterTrace], Learning, list[ISpecification], int, float]]]:
        trace, cts, learning_type, spec_history, learning_steps, learning_time = data
        return self._hm.select_alternative_learning_tasks(self._mitigation_strategies[learning_type](spec, trace, cts, spec_history, learning_steps, learning_time))


    def prepare_learning_task(
            self,
            spec: SpectraSpecification,
            data: Tuple[list[str], list[CounterTrace], Learning, list[SpectraSpecification], int, float],
            learned_spec: SpectraSpecification,
            counter_argument
    ) -> Tuple[ISpecification, Tuple[list[str], list[CounterTrace], Learning, list[ISpecification], int, float]]:
        trace, cts, learning_type, spec_history, learning_steps, learning_time = data
        if learning_type == Learning.ASSUMPTION_WEAKENING:
            return spec, (trace, cts + [counter_argument], learning_type, spec_history + [deepcopy(learned_spec)], learning_steps, learning_time)
        else:
            return learned_spec, (trace, cts + [counter_argument], learning_type, spec_history + [deepcopy(learned_spec)], learning_steps, learning_time)
