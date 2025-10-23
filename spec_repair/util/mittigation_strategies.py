import re
from copy import deepcopy, copy
from typing import List, Tuple, Set, Any, cast

from spec_repair.components.interfaces.ispecification import ISpecification
from spec_repair.components.new_spec_encoder import NewSpecEncoder
from spec_repair.enums import Learning
from spec_repair.exceptions import NoViolationException
from spec_repair.helpers.counter_trace import CounterTrace, complete_cts_from_ct
from spec_repair.helpers.spectra_specification import SpectraSpecification
from spec_repair.wrappers.asp_wrappers import get_violations


def move_one_to_guarantee_weakening(
        spec: ISpecification,  # ignored
        trace: list[str],
        cts: List[CounterTrace],
        spec_history: List[ISpecification],
        learning_steps: int,
        learning_time: float
) -> List[Tuple[
    ISpecification, Tuple[list[str], list[CounterTrace], Learning, list[ISpecification], int, float]]]:
    new_spec = spec_history[0]
    new_cts = cts[0:1]  # Only keep the first counter-trace
    new_learning_type = Learning.GUARANTEE_WEAKENING
    new_spec_history = []
    new_data = (trace, new_cts, new_learning_type, new_spec_history, learning_steps, learning_time)
    return [(new_spec, new_data)]

def move_all_to_guarantee_weakening(
        spec: ISpecification,  # ignored
        trace: list[str],
        cts: List[CounterTrace],
        spec_history: List[ISpecification],
        learning_steps: int,
        learning_time: float
) -> List[Tuple[
    ISpecification, Tuple[list[str], list[CounterTrace], Learning, list[ISpecification], int, float]]]:
    new_spec_cts_pairs = zip(spec_history, cts)
    new_learning_type = Learning.GUARANTEE_WEAKENING
    new_spec_history = []
    new_tasks = [(new_spec, (trace, [new_cts], new_learning_type, deepcopy(new_spec_history), learning_steps, learning_time)) for new_spec, new_cts in new_spec_cts_pairs]
    return new_tasks


def complete_counter_traces(
        spec: ISpecification,
        trace: list[str],
        cts: List[CounterTrace],
        spec_history: List[ISpecification],
        learning_steps: int,
        learning_time: float
) -> List[Tuple[
    ISpecification, Tuple[list[str], list[CounterTrace], Learning, list[ISpecification], int, float]]]:
    ctss: Set[Tuple[CounterTrace, ...]] = {tuple(cts)}
    unchanged = False
    while not unchanged:
        unchanged = True
        for cts in deepcopy(ctss):
            asp: str = NewSpecEncoder.encode_ASP(cast(SpectraSpecification, spec), trace, list(cts))
            violations = get_violations(asp, exp_type=Learning.GUARANTEE_WEAKENING.exp_type())
            if not violations:
                raise NoViolationException("Violation trace is not violating!")
            deadlock_required = re.findall(r"entailed\((counter_strat_\d*_\d*)\)", ''.join(violations))
            if deadlock_required:
                set_cts = set(cts)
                for i, ct in enumerate(copy(cts)):
                    if ct.is_deadlock() and ct.get_name() in deadlock_required:
                        new_set_cts = copy(set_cts)
                        new_set_cts.remove(ct)
                        ctss |= set([tuple(new_set_cts | {complete_ct}) for complete_ct in
                                     complete_cts_from_ct(ct, spec.to_str().split('\n'), deadlock_required)])
                        unchanged = False
                if not unchanged:
                    ctss.remove(cts)
    possible_cts_list = [list(cts) for cts in ctss]
    alternative_learning_tasks: List[Tuple[ISpecification, Any]] = []
    for possible_cts in possible_cts_list:
        new_spec = deepcopy(spec)
        new_learning_type = Learning.GUARANTEE_WEAKENING
        new_data = (trace, possible_cts, new_learning_type, deepcopy(spec_history), learning_steps, learning_time)
        alternative_learning_tasks.append((new_spec, new_data))
    return alternative_learning_tasks
