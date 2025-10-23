from abc import ABC, abstractmethod
from typing import Any

from spec_repair.components.interfaces.ispecification import ISpecification


class IDiscriminator(ABC):
    @abstractmethod
    def get_learning_strategy(
            self,
            spec: ISpecification,
            data: Any
    ) -> str:
        """
        Given a specification and data, return the learning strategy.
        :param spec: The specification.
        :param data: The data to learn from.
        :return: The learning strategy.
        """
        pass
