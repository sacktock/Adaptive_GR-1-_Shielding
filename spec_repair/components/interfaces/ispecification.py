from abc import ABC, abstractmethod


class ISpecification(ABC):
    @abstractmethod
    def to_str(self) -> str:
        """
        Convert the specification to a string representation.
        """
        pass
