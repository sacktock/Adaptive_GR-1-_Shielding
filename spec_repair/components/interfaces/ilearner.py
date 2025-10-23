from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from spec_repair.components.interfaces.ispecification import ISpecification


class ILearner(ABC):
    @abstractmethod
    def learn_new(
        self,
        spec: ISpecification,
        data: Any
    ) -> List[Tuple[ISpecification, Any]]:
        """
        Given a specification and data, learn new specifications.
        :param spec: The original specification.
        :param data: The data to learn from.
        :return: A list of new specifications.
        """
        pass
