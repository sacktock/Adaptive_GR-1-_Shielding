import re
from typing import Optional, Dict

from spec_repair.enums import Learning
from spec_repair.builders.abstract_builder import AbstractBuilder
from spec_repair.builders.enums import ChoiceType


class ChoicesBuilder(AbstractBuilder):

    def __init__(self):
        self.recording: bool = False
        self.choice: int = -1
        self.choice_type: Optional[ChoiceType] = None
        self.options: Dict[int, str] = dict()
        self._index: int = -1

    def start(self):
        self.recording: bool = True
        self.choice: int = -1
        self.choice_type: Optional[ChoiceType] = None
        self.options: Dict[int, str] = dict()
        self._index: int = -1

    def record(self, line: str) -> bool:
        if not line.strip():
            # Skip newlines
            return self.recording
        if not self.recording:
            raise ValueError(f"Rule builder finished recording! This should never be reached!")
        match = re.match(r"^Choice taken: '(\d+)'$", line)
        if match:
            self.choice = int(match.group(1))
            self.recording = False
            return self.recording
        match = re.match(r'(\d+):\s*(.*)', line)
        if match:
            if '[' in line and '(' in line:
                self.choice_type = ChoiceType.EXP_WEAKENING
            elif '(' in line:
                self.choice_type = ChoiceType.CS_GENERATION
            else:
                self.choice_type = ChoiceType.CS_DEADLOCK_COMPLETION
            self._index = int(match.group(1))
            content = match.group(2)
            self.options[self._index] = content
        elif (not re.match("Enter the index of your choice \[\d+-\d+]", line)
              and not re.match("^Select an option by choosing its index:$", line)):
            self.options[self._index] += f"\n{line.strip()}"
        return self.recording

    def to_string(self, learning: Learning):
        """
        e.g. output:
GW: 1
0: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(highwater,V1,V2).', '', '']
1: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(methane,V1,V2).', '', '']
2: ['consequent_exception(guarantee1_1,V1,V2) :- holds_at(pump,V1,V2).', '', '']
        """
        match learning:
            case Learning.ASSUMPTION_WEAKENING:
                l = 'A'
            case Learning.GUARANTEE_WEAKENING:
                l = 'G'

        match self.choice_type:
            case ChoiceType.EXP_WEAKENING:
                ch = 'W'
            case ChoiceType.CS_GENERATION:
                ch = 'CS'
            case ChoiceType.CS_DEADLOCK_COMPLETION:
                ch = 'D'

        opts = "\n".join([f"{i}: {opt}" for i, opt in self.options.items()])

        return f"{l}{ch}: {self.choice}\n{opts}"
