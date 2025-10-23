from enum import Enum


class ChoiceType(Enum):
    EXP_WEAKENING = 1
    CS_GENERATION = 2
    CS_DEADLOCK_COMPLETION = 3
    OTHER = 4


class RecordingState(Enum):
    NOTHING = 0
    CHOICES = 1
    RULE_CHANGE = 2
    SUMMARY = 3


class RuleToExpect(Enum):
    OLD_RULE = 0
    HYPOTHESIS = 1
    NEW_RULE = 2


class Rule:
    name: str
    old: str
    new: str

    def __init__(self, name="", old="", new=""):
        self.name = name
        self.old = old
        self.new = new
