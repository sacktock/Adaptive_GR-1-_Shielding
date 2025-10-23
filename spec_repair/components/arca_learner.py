import re
from copy import deepcopy
from typing import List, Tuple, Optional

from spec_repair.components.interfaces.ilearner import ILearner
from spec_repair.components.new_spec_encoder import NewSpecEncoder
from spec_repair.helpers.adaptation_learned import Adaptation
from spec_repair.helpers.counter_trace import CounterTrace, complete_cts_from_ct
from spec_repair.enums import Learning
from spec_repair.exceptions import NoViolationException, NoWeakeningException, DeadlockRequiredException, \
    NoAssumptionWeakeningException
from spec_repair.helpers.heuristic_managers.iheuristic_manager import IHeuristicManager
from spec_repair.helpers.heuristic_managers.no_filter_heuristic_manager import NoFilterHeuristicManager
from spec_repair.helpers.ilasp_interpreter import ILASPInterpreter
from spec_repair.helpers.spectra_specification import SpectraSpecification

from spec_repair.wrappers.asp_wrappers import get_violations, run_ILASP


class ARCALearner(ILearner):
    def __init__(
            self,
            heuristic_manager: IHeuristicManager = NoFilterHeuristicManager(),
    ):
        self._hm = heuristic_manager
        self.spec_encoder = NewSpecEncoder(heuristic_manager)

    # TODO: consider returning "data" instead of empty list when no learning is possible
    def learn_new(
            self,
            spec: SpectraSpecification,
            data: Tuple[list[str], list[CounterTrace], Learning, list[SpectraSpecification], int, float]
    ) -> List[Tuple[SpectraSpecification, Tuple[list[str], list[CounterTrace], Learning, list[SpectraSpecification], int, float]]]:
        trace, cts, learning_type, spec_history, learning_steps, learning_time = data
        assert learning_type == Learning.ASSUMPTION_WEAKENING
        try:
            possible_adaptations: List[List[Adaptation]] = self.find_possible_adaptations(spec, trace, cts, learning_type)
            if self._hm:
                possible_adaptations = self._hm.select_possible_learning_adaptations(possible_adaptations)
            new_specs = [deepcopy(spec).integrate_multiple(adaptations) for adaptations in possible_adaptations]
            # Moves straight to guarantee weakening after the first try
            new_data = (trace, cts, Learning.GUARANTEE_WEAKENING, spec_history, learning_steps + 1, learning_time)
            new_tasks = [(new_spec, deepcopy(new_data)) for new_spec in new_specs]
            return new_tasks
        except NoWeakeningException as e:
            print(f"Weakening failed: {e}")
            return []
        except NoViolationException as e:
            print(f"Weakening failed: {e}")
            return []
        except DeadlockRequiredException as e:
            print(f"Weakening failed: {e}")
            return []

    def find_possible_adaptations(self, spec: SpectraSpecification, trace, cts, learning_type) -> List[
        List[Adaptation]]:
        violations = self.get_spec_violations(spec, trace, cts, learning_type)
        ant_adaptations = self.find_antecedent_exception_adaptations(spec, trace, cts, learning_type, violations)
        con_adaptations = self.find_consequent_exception_adaptations(spec, trace, cts, learning_type, violations)
        ev_adaptations = self.find_eventualisation_adaptations(spec, trace, cts, learning_type, violations)
        adaptations = ant_adaptations + con_adaptations + ev_adaptations
        # adaptations = self.find_all_exception_adaptations(spec, trace, cts, learning_type, violations)
        if not adaptations:
            if learning_type == Learning.ASSUMPTION_WEAKENING:
                raise NoAssumptionWeakeningException(
                    f"No {learning_type.exp_type_str()} weakening produces realizable spec (las file UNSAT)"
                )
            else:
                raise NoWeakeningException(
                    f"No {learning_type.exp_type_str()} weakening produces realizable spec (las file UNSAT)")

        useful_adaptations: List[List[Adaptation]] = filter_useful_adaptations(adaptations)
        return useful_adaptations

    def find_all_exception_adaptations(self, spec, trace, cts, learning_type, violations) -> List[Tuple[int, List[Adaptation]]]:
        hm = NoFilterHeuristicManager()
        hm.set_enabled("ANTECEDENT_WEAKENING")
        hm.set_enabled("CONSEQUENT_WEAKENING")
        hm.set_enabled("INVARIANT_TO_RESPONSE_WEAKENING")
        return self.find_adaptations_with_heuristic(spec, trace, cts, learning_type, violations, hm)

    def find_antecedent_exception_adaptations(self, spec, trace, cts, learning_type, violations) -> List[Tuple[int, List[Adaptation]]]:
        hm = NoFilterHeuristicManager()
        hm.set_enabled("ANTECEDENT_WEAKENING")
        hm.set_disabled("CONSEQUENT_WEAKENING")
        hm.set_disabled("INVARIANT_TO_RESPONSE_WEAKENING")
        return self.find_adaptations_with_heuristic(spec, trace, cts, learning_type, violations, hm)

    def find_consequent_exception_adaptations(self, spec, trace, cts, learning_type, violations) -> List[Tuple[int, List[Adaptation]]]:
        hm = NoFilterHeuristicManager()
        hm.set_disabled("ANTECEDENT_WEAKENING")
        hm.set_enabled("CONSEQUENT_WEAKENING")
        hm.set_disabled("INVARIANT_TO_RESPONSE_WEAKENING")
        return self.find_adaptations_with_heuristic(spec, trace, cts, learning_type, violations, hm)

    def find_eventualisation_adaptations(self, spec, trace, cts, learning_type, violations) -> List[Tuple[int, List[Adaptation]]]:
        hm = NoFilterHeuristicManager()
        hm.set_disabled("ANTECEDENT_WEAKENING")
        hm.set_disabled("CONSEQUENT_WEAKENING")
        hm.set_enabled("INVARIANT_TO_RESPONSE_WEAKENING")
        return self.find_adaptations_with_heuristic(spec, trace, cts, learning_type, violations, hm)

    def find_adaptations_with_heuristic(self, spec, trace, cts, learning_type, violations, hm):
        self.spec_encoder.set_heuristic_manager(hm)
        ilasp: str = self.spec_encoder.encode_ILASP(spec, trace, cts, violations, learning_type)
        output: str = run_ILASP(ilasp)
        adaptations: Optional[
            List[Tuple[int, List[Adaptation]]]] = ILASPInterpreter.extract_learned_possible_adaptations(output)
        if not adaptations:
            return []
        return adaptations

    def get_spec_violations(self, spec: SpectraSpecification, trace, cts, learning_type) -> List[str]:
        asp: str = self.spec_encoder.encode_ASP(spec, trace, cts)
        violations = get_violations(asp, exp_type=learning_type.exp_type())
        if not violations:
            raise NoViolationException("Violation trace is not violating!")
        if learning_type == Learning.GUARANTEE_WEAKENING:
            deadlock_required = re.findall(r"entailed\((counter_strat_\d*_\d*)\)", ''.join(violations))
            violation_ct = re.findall(r"violation_holds\([^,]*,[^,]*,\s*(counter_strat_\d+_\d+)", ''.join(violations))
            if deadlock_required and not violation_ct:
                raise DeadlockRequiredException("Violation trace is not violating! Deadlock completion is required.")
        return violations


def filter_useful_adaptations(potential_adaptations: List[Tuple[int, List[Adaptation]]]) -> List[List[Adaptation]]:
    ev_adaptations = [(score, adaptations) for score, adaptations in potential_adaptations if
                      all(adaptation.type == "ev_temp_op" for adaptation in adaptations)]
    other_adaptations = [(score, adaptations) for score, adaptations in potential_adaptations if
                         (score, adaptations) not in ev_adaptations]
    top_adaptations = ([adaptations for score, adaptations in other_adaptations if
                        score == min(other_adaptations, key=lambda x: x[0])[0]] +
                       [adaptations for score, adaptations in ev_adaptations if
                        score == min(ev_adaptations, key=lambda x: x[0])[0]])
    return top_adaptations
