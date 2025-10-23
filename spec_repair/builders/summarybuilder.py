import re
from typing import Optional

from spec_repair.builders.abstract_builder import AbstractBuilder


class SummaryBuilder(AbstractBuilder):
    def __init__(self):
        self.spec_id: Optional[int] = None
        self.solution_status: str = ""
        self.choices: list[int] = []
        self.recording: bool = False

    def start(self):
        self.spec_id: Optional[int] = None
        self.solution_status: str = ""
        self.choices: list[int] = []
        self.recording: bool = True

    def record(self, line: str) -> bool:
        if not line.strip():
            # Skip newlines
            return self.recording
        if not self.recording:
            raise ValueError(f"Summary builder finished recording! This should never be reached!")
        match = re.match(r'\b\w+\s+run ended with \[([\d\s,]+)\]', line)
        if match:
            self.choices = [int(x) for x in match.group(1).split(',')]
            return self.recording
        match = re.search(r'SPEC ID: (\d+)\.', line)
        if match:
            self.spec_id = int(match.group(1))
            return self.recording
        match = re.match(r'This run is\s+(.+)', line)
        if match:
            self.solution_status = match.group(1)
        elif re.match(r'^Moving to next run \[\d+(?:, \d+)*\]$', line):
            pass
        elif (line.strip() == '#########################'
              and self.solution_status != ""):
            self.recording = False
        else:
            self.solution_status += f" {line.strip()}"

        return self.recording
