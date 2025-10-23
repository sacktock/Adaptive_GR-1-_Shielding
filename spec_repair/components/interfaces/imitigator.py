from abc import ABC, abstractmethod
from typing import List, Tuple, Any

from spec_repair.components.interfaces.ispecification import ISpecification


class IMitigator(ABC):
    @abstractmethod
    def prepare_alternative_learning_tasks(self, spec, data) -> List[Tuple[ISpecification, Any]]:
        pass

    @abstractmethod
    def prepare_learning_task(self, spec, data, learned_spec, counter_argument) -> Tuple[ISpecification, Any]:
        pass
