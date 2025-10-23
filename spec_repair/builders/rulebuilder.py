import re
from typing import Optional, List

from spec_repair.builders.enums import Rule, RuleToExpect


class RuleBuilder:
    def __init__(self):
        self.cur_rule: Optional[Rule] = None
        self.rules: List[Rule] = []
        self.recording: Optional[RuleToExpect] = None

    def start(self):
        self.cur_rule = Rule()
        self.rules = []
        self.recording = RuleToExpect.OLD_RULE

    def record(self, line: str) -> bool:
        """
        :param line:
        :return: continues: Whether the rule should continue recording, or it reached an end line.
        """
        if not line.strip():
            # Skip newlines
            return self.recording is not None
        if not self.recording:
            raise ValueError(f"Rule builder finished recording! This should never be reached!")
        found_eventually = re.match(r'^\s*([^:]+):\s*$', line)
        if found_eventually:
            keyword = found_eventually.group(1)
            match keyword:
                case "Rule":
                    self.recording = RuleToExpect.OLD_RULE
                    self.cur_rule = Rule()
                case "Hypothesis":
                    self.recording = RuleToExpect.HYPOTHESIS
                case "New Rule":
                    self.recording = RuleToExpect.NEW_RULE
                case _:
                    raise ValueError(f"Invalid keyword found: '{keyword}'.")
        elif re.match(r'^(Unrealizable|Realizable: success.)$', line):
            self.rule = None
            self.recording = None
        elif line:
            match self.recording:
                case RuleToExpect.OLD_RULE:
                    self.cur_rule.old = line.strip()
                case RuleToExpect.HYPOTHESIS:
                    baseline_weakening_pattern = r'(consequent|antecedent)_exception\(([^,]+),'
                    eventually_weakening_pattern = r'consequent_holds\(([^,]+),([^,]+),([^,]+),([^,]+)\)'
                    found_baseline = re.search(baseline_weakening_pattern, line)
                    found_eventually = re.search(eventually_weakening_pattern, line)
                    if found_baseline:
                        self.cur_rule.name = found_baseline.group(2)
                    elif found_eventually:
                        self.cur_rule.name = found_eventually.group(2)
                    else:
                        raise ValueError(f"The line '{line}' does not fit the hypothesis structure.")
                case RuleToExpect.NEW_RULE:
                    self.cur_rule.new = line.strip()
                    self.recording = RuleToExpect.OLD_RULE
                    self.rules.append(self.cur_rule)
                    self.cur_rule = Rule()
                case _:
                    raise NotImplementedError()
        return self.recording is not None

    def __str__(self) -> str:
        return '\n'.join([f"{rule.name}:\n{rule.old}\n=>\n{rule.new}" for rule in self.rules])



