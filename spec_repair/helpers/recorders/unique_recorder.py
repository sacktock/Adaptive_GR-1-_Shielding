from typing import Optional, List

from spec_repair.helpers.recorders.irecorder import IRecorder
from spec_repair.heuristics import T


class UniqueRecorder(IRecorder[T]):
    def __init__(self):
        self._set: set[T] = set()
        self._value_to_id: dict[T, int] = {}
        self._next_id: int = 0

    def _deep_freeze(self, obj):
        if isinstance(obj, list):
            return tuple(self._deep_freeze(item) for item in obj)
        elif isinstance(obj, dict):
            return tuple((self._deep_freeze(k), self._deep_freeze(v)) for k, v in obj.items())
        elif isinstance(obj, set):
            return frozenset(self._deep_freeze(item) for item in obj)
        elif isinstance(obj, tuple):
            return tuple(self._deep_freeze(item) for item in obj)
        else:
            return obj

    def add(self, value: T) -> int:
        """
        Adds an element to the set if it's not already present and assigns it a unique ID.
        Returns the ID of the element.
        :param value: The element to add.
        :return: The ID of the element.
        """
        value = self._deep_freeze(value)
        if value not in self._set:
            self._set.add(value)
            self._value_to_id[value] = self._next_id
            self._next_id += 1
        return self._value_to_id[value]

    def get_id(self, value: T) -> Optional[int]:
        """
        Returns the ID associated with the element, or None if the element is not present.
        :param value: The element to get the ID for.
        :return: The ID of the element, or None if the element is not present.
        """
        value = self._deep_freeze(value)
        return self._value_to_id.get(value)

    def get_element_by_id(self, id_: int) -> Optional[T]:
        """
        Returns the element associated with the given ID, or None if no such element exists.
        :param id_: The ID to get the element for.
        :return: The element associated with the ID, or None if no such element exists.
        """
        for elem, elem_id in self._value_to_id.items():
            if elem_id == id_:
                return elem
        return None

    def get_all_values(self) -> List[T]:
        """
        Returns a list of all unique elements stored in the set.
        :return: A list of all unique elements stored in the set.
        """
        return list(self._set)

    def __contains__(self, element: T) -> bool:
        """
        Checks if an element is in the ElementManager.
        Returns True if the element is present, otherwise False.
        :param element: The element to check for.
        :return: True if the element is present, otherwise False.
        """
        element = self._deep_freeze(element)
        return element in self._set
