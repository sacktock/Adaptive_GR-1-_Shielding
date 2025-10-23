from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional

T = TypeVar("T")


class IRecorder(ABC, Generic[T]):
    @abstractmethod
    def add(self, value: T) -> int:
        pass

    @abstractmethod
    def get_id(self, value: T) -> Optional[int]:
        pass

    @abstractmethod
    def get_element_by_id(self, id_: int) -> Optional[T]:
        pass

    @abstractmethod
    def get_all_values(self) -> List[T]:
        pass

    @abstractmethod
    def __contains__(self, element: T) -> bool:
        pass