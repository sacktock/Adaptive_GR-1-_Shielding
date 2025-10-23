from abc import ABC, abstractmethod


class IOracle(ABC):
    @abstractmethod
    def is_valid_or_counter_arguments(self, new_spec, data):
        pass
