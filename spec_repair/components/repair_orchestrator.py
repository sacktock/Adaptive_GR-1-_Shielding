from typing import Optional

from spec_repair.helpers.counter_trace import CounterTrace, ct_from_cs
from spec_repair.components.spec_learner import SpecLearner, select_learning_hypothesis
from spec_repair.components.spec_oracle import SpecOracle
from spec_repair.enums import Learning
from spec_repair.exceptions import NoWeakeningException
from spec_repair.heuristics import manual_choice, choose_one_with_heuristic
from spec_repair.ltl_types import CounterStrategy
from spec_repair.special_types import StopHeuristicType


class RepairOrchestrator:
    def __init__(self, learner: SpecLearner, oracle: SpecOracle):
        self._learner = learner
        self._oracle = oracle
        self._ct_cnt = 0

    # Reimplementation of the highest level abstraction code
    def repair_spec(
            self,
            spec: list[str],
            trace: list[str],
            stop_heuristic:
            StopHeuristicType = lambda a, g: True
    ) -> list[str]:
        self._ct_cnt = 0
        ct_asm, ct_gar = [], []
        weak_spec_history = []

        # Assumption Weakening for Consistency
        weak_spec: list[str] = self._learner.learn_weaker_spec(
            spec, trace, list(),
            learning_type=Learning.ASSUMPTION_WEAKENING,
            heuristic=manual_choice)
        weak_spec_history.append(weak_spec)
        cs: Optional[CounterStrategy] = self._oracle.synthesise_and_check(weak_spec)

        # Assumption Weakening for Realisability
        try:
            while cs:  # not is_realisable
                ct_asm.append(self.ct_from_cs(cs))
                weak_spec: list[str] = self._learner.learn_weaker_spec(
                    spec, trace, ct_asm,
                    learning_type=Learning.ASSUMPTION_WEAKENING,
                    heuristic=manual_choice)
                if weak_spec == spec and stop_heuristic(spec, ct_asm):
                    break
                weak_spec_history.append(weak_spec)
                cs = self._oracle.synthesise_and_check(weak_spec)
        except NoWeakeningException as e:
            print(str(e))

        if not cs:
            return weak_spec_history[-1]
        print("Moving to Guarantee Weakening")

        # Guarantee Weakening
        spec = weak_spec_history[0]
        ct_gar.append(ct_asm[0])
        while cs:
            ct_gar = self.complete_counter_traces(spec, trace, ct_gar)
            hypotheses = self._learner.find_weakening_hypotheses(spec, trace, ct_gar, Learning.GUARANTEE_WEAKENING)
            learning_hypothesis = select_learning_hypothesis(hypotheses, manual_choice)
            spec: list[str] = self._learner.integrate_learning_hypothesis(spec, learning_hypothesis, Learning.GUARANTEE_WEAKENING)
            cs = self._oracle.synthesise_and_check(spec)
            if cs:
                ct_gar.append(self.ct_from_cs(cs))

        return spec

    def complete_counter_traces(self, spec: list[str], trace: list[str], ct_gar: list[CounterTrace]) -> list[CounterTrace]:
        complete_cts = self._learner.get_all_complete_counter_trace_lists(
            spec, trace, ct_gar, Learning.GUARANTEE_WEAKENING)
        if not complete_cts:
            raise NoWeakeningException("No weakening found")
        return choose_one_with_heuristic(complete_cts, manual_choice)

    def ct_from_cs(self, cs: list[str]) -> CounterTrace:
        ct = ct_from_cs(cs, heuristic=manual_choice, cs_id=self._ct_cnt)
        self._ct_cnt += 1
        return ct
