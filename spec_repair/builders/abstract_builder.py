from abc import ABC, abstractmethod


class AbstractBuilder(ABC):
    @abstractmethod
    def record(self, line: str) -> bool:
        pass
