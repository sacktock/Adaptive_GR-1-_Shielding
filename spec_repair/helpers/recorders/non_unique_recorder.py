from copy import deepcopy
from typing import Optional, List

from spec_repair.helpers.recorders.irecorder import IRecorder
from spec_repair.heuristics import T


class NonUniqueRecorder(IRecorder[T]):
    def __init__(self):
        self._list: list[T] = []

    def add(self, value: T) -> int:
        """
        Adds an element to the set if it's not already present and assigns it a unique ID.
        Returns the ID of the element.
        :param value: The element to add.
        :return: The ID of the element.
        """
        self._list.append(value)
        return len(self._list) - 1

    def get_id(self, value: T) -> Optional[int]:
        """
        Returns the ID associated with the element, or None if the element is not present.
        :param value: The element to get the ID for.
        :return: The ID of the element, or None if the element is not present.
        """
        raise NotImplementedError("NonUniqueRecorder does not support get_id")

    def get_element_by_id(self, id_: int) -> Optional[T]:
        """
        Returns the element associated with the given ID, or None if no such element exists.
        :param id_: The ID to get the element for.
        :return: The element associated with the ID, or None if no such element exists.
        """
        if 0 <= id_ < len(self._list):
            return self._list[id_]
        return None

    def get_all_values(self) -> List[T]:
        """
        Returns a list of all unique elements stored in the set.
        :return: A list of all unique elements stored in the set.
        """
        return deepcopy(self._list)

    def __contains__(self, element: T) -> bool:
        """
        Checks if an element is in the ElementManager.
        Returns True if the element is present, otherwise False.
        :param element: The element to check for.
        :return: True if the element is present, otherwise False.
        """
        return element in self._list
