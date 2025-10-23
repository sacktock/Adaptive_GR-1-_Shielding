import csv
import os
from abc import ABC, abstractmethod


class Logger(ABC):
    """
    An abstract class for logging
    """

    @abstractmethod
    def log_transition(self, t_from: int, t_to: int, learning_type: str):
        pass


class NoLogger(Logger):
    """
    A logger that does nothing.
    """

    def log_transition(self, t_from: int, t_to: int, learning_type: str):
        pass


class RepairLogger(Logger):
    def __init__(self, transitions_file_path: str, debug: bool = False):
        self._transitions_file_path = transitions_file_path
        self._debug = debug

        # Create the file with headers if it doesn't exist
        if not os.path.isfile(self._transitions_file_path):
            with open(self._transitions_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["from", "to", "type"])

        # Open the file in append mode
        self.file = open(self._transitions_file_path, mode='a', newline='')
        self.writer = csv.writer(self.file)

    def log_transition(self, from_node: int, to_node: int, learning_type: str):
        self.writer.writerow([from_node, to_node, learning_type])

        # If debug mode is on, flush the file to make sure it is updated immediately
        if self._debug:
            self.file.flush()
