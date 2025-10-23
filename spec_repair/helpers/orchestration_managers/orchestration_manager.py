from collections import deque
from typing import Deque, Tuple, Any

from spec_repair.components.interfaces.ispecification import ISpecification
from spec_repair.helpers.recorders.irecorder import IRecorder
from spec_repair.helpers.recorders.unique_recorder import UniqueRecorder


# TODO: turn this one into an argument in the BFS strategy initialisation as well
class OrchestrationManager:
    def __init__(self):
        self._stack: Deque[Tuple[ISpecification, Any]] = deque()
        self._visited_nodes: IRecorder[Tuple[ISpecification, Any]] = UniqueRecorder()

    def _reset(self):
        self._stack.clear()
        self._visited_nodes = UniqueRecorder()

    def initialise_learning_tasks(self, spec: ISpecification, data: Any):
        self._reset()
        self.enqueue_new_tasks(spec, data)

    def enqueue_new_tasks(self, spec: ISpecification, data: Any):
        node: Tuple[Any, Any] = (spec, data)
        self._stack.append(node)
        self._visited_nodes.add(node)

    def has_next(self) -> bool:
        return bool(self._stack)

    def get_next(self) -> Tuple[ISpecification, Any]:
        return self._stack.popleft()
