from datetime import datetime
from typing import Any, Dict, List, Tuple

from spec_repair.components.interfaces.idiscriminator import IDiscriminator
from spec_repair.components.interfaces.ilearner import ILearner
from spec_repair.components.interfaces.imitigator import IMitigator
from spec_repair.components.interfaces.ioracle import IOracle
from spec_repair.components.interfaces.ispecification import ISpecification
from spec_repair.helpers.heuristic_managers.iheuristic_manager import IHeuristicManager
from spec_repair.helpers.heuristic_managers.no_filter_heuristic_manager import NoFilterHeuristicManager
from spec_repair.helpers.orchestration_managers.orchestration_manager import OrchestrationManager
from spec_repair.helpers.recorders.irecorder import IRecorder
from spec_repair.helpers.recorders.unique_recorder import UniqueRecorder


class SpecLogger:
    def __init__(self, filename: str = "spec_repair.log"):
        self.filename = filename
        with open(self.filename, 'a') as f:
            f.write(f"[SpecLogger] Started at: {datetime.now()}\n")

    def record(self, idx: int, spec: ISpecification, data: Any):
        trace, cts, learning_type, spec_history, learning_steps, learning_time = data
        log_message = f"[SpecLogger] Index: {idx}, learning_type: {learning_type}, learning_steps: {learning_steps}, learning_time: {learning_time}\n"
        with open(self.filename, 'a') as f:
            f.write(log_message)


class BFSRepairOrchestrator:
    def __init__(
            self,
            learners: Dict[str, ILearner],
            oracle: IOracle,
            discriminator: IDiscriminator,
            mitigator: IMitigator,
            heuristic_manager: IHeuristicManager = NoFilterHeuristicManager(),
            recorder: IRecorder[ISpecification] = UniqueRecorder(),
            logger: SpecLogger = SpecLogger("spec_repair.log")
    ):
        self._learners = learners
        self._oracle = oracle
        self._discriminator = discriminator
        self._mitigator = mitigator
        self._om = OrchestrationManager()
        self._hm = heuristic_manager
        self._recorder = recorder
        self._logger = logger
        self._initialise_repair()

    def _initialise_repair(self):
        # Counter for recording counter-traces
        self._ct_cnt = 0
        self._hm.reset()
        for learner in self._learners.values():
            learner._hm = self._hm
        self._mitigator._hm = self._hm
        self._oracle._hm = self._hm

    def repair_bfs(
            self,
            og_spec: ISpecification,
            og_data: Any
    ):
        self._initialise_repair()
        self._om.initialise_learning_tasks(og_spec, og_data)

        while self._om.has_next():
            spec, data = self._om.get_next()
            learning_strategy: str = self._discriminator.get_learning_strategy(spec, data)
            learner = self._learners[learning_strategy]
            learned_tasks: List[Tuple[ISpecification, Any]] = learner.learn_new(spec, data)
            if not learned_tasks:
                alt_tasks: List[Tuple[ISpecification, Any]] = self._mitigator.prepare_alternative_learning_tasks(spec,
                                                                                                                 data)
                for alt_spec, alt_data in alt_tasks:
                    self._om.enqueue_new_tasks(alt_spec, alt_data)
            else:
                for learned_spec, data in learned_tasks:
                    counter_examples_with_data: List[Tuple[Any, Any]] = self._oracle.is_valid_or_counter_arguments(
                        learned_spec, data)
                    if not counter_examples_with_data:
                        learned_id = self._recorder.add(learned_spec)
                        self._logger.record(learned_id, learned_spec, data)
                    else:
                        # TODO: find a way to filter the counter examples using the heuristic manager
                        for counter_example, data in counter_examples_with_data:
                            new_spec, new_data = self._mitigator.prepare_learning_task(spec, data, learned_spec,
                                                                                       counter_example)
                            self._om.enqueue_new_tasks(new_spec, new_data)
