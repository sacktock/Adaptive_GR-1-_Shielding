import copy
import re
from typing import Any

import pandas as pd
import os

from spec_repair.enums import Learning
from spec_repair.builders.abstract_builder import AbstractBuilder
from spec_repair.builders.choicesbuilder import ChoicesBuilder
from spec_repair.builders.enums import RecordingState, ChoiceType
from spec_repair.builders.rulebuilder import RuleBuilder
from spec_repair.helpers.recorders.unique_spec_recorder import UniqueSpecRecorder
from spec_repair.builders.summarybuilder import SummaryBuilder
from spec_repair.util.file_util import generate_temp_filename
from spec_repair.wrappers.spec import Spec


class CSVBuilder(AbstractBuilder):
    def __init__(self):
        self.columns = [
            "choices",
            "assumption weakening steps",
            "assumption counter-strategy generation steps",
            "guarantee weakening steps",
            "guarantee counter-strategy generation steps",
            "guarantee deadlock generations",
            "solution status compared to ideal",
            "id solution reached",
            "assumptions modified",
            "guarantees modified",
            "rules and modifications",
            "cleaned output (choices and results)"
        ]
        self.dtype_mapping = {
            "choices": list[str],
            "assumption weakening steps": int,
            "assumption counter-strategy generation steps": int,
            "guarantee weakening steps": int,
            "guarantee counter-strategy generation steps": int,
            "guarantee deadlock generations": int,
            "solution status compared to ideal": str,
            "id solution reached": int,
            "assumptions modified": int,
            "guarantees modified": int,
            "rules and modifications": str,
            "cleaned output (choices and results)": str
        }
        # Initialize the DataFrame with columns
        self._df = pd.DataFrame(columns=list(self.dtype_mapping.keys()))

        # Reset state for new column
        # self.add_empty_row()
        self.rule_builder = RuleBuilder()
        self.choices_builder = ChoicesBuilder()
        self.summary_builder = SummaryBuilder()
        self.spec_recorder = UniqueSpecRecorder()
        self.ideal_spec = None
        self._reset_builder_for_new_row()

    def add_ideal_spec(self, ideal_spec: Spec):
        assert len(self.spec_recorder.get_all_values()) == 0
        self.spec_recorder.add(ideal_spec)
        self.ideal_spec = ideal_spec
        self.cur_spec = copy.deepcopy(ideal_spec)

    def _reset_builder_for_new_row(self):
        self.recording_state = RecordingState.NOTHING
        self.phase: Learning = Learning.ASSUMPTION_WEAKENING
        self.num_solutions: int = 0
        self.cleaned_output: str = ""
        self.rules_and_modifications: list[str] = []
        self.cur_spec = copy.deepcopy(self.ideal_spec)
        self.add_empty_row()

    def add_empty_row(self):
        self._df.loc[len(self._df)] = [None] * len(self.columns)

    def record(self, line: str) -> bool:
        # Skip empty lines
        if not line.strip():
            return False
        self.assert_line_for_errors(line)
        match self.recording_state:
            case RecordingState.CHOICES:
                continues: bool = self.choices_builder.record(line)
                if not continues:
                    self.recording_state = RecordingState.NOTHING
                    self.cleaned_output += f"\n\n{self.choices_builder.to_string(self.phase)}"
                    match self.choices_builder.choice_type:
                        case ChoiceType.EXP_WEAKENING:
                            match self.phase:
                                case Learning.ASSUMPTION_WEAKENING:
                                    self._increment_in_table("assumption weakening steps")
                                case Learning.GUARANTEE_WEAKENING:
                                    self._increment_in_table("guarantee weakening steps")
                        case ChoiceType.CS_GENERATION:
                            match self.phase:
                                case Learning.ASSUMPTION_WEAKENING:
                                    self._increment_in_table("assumption counter-strategy generation steps")
                                case Learning.GUARANTEE_WEAKENING:
                                    self._increment_in_table("guarantee counter-strategy generation steps")
                        case ChoiceType.CS_DEADLOCK_COMPLETION:
                            self._increment_in_table("guarantee deadlock generations")
            case RecordingState.RULE_CHANGE:
                continues: bool = self.rule_builder.record(line)
                if not continues:
                    self.recording_state = RecordingState.NOTHING
                    match self.phase:
                        case Learning.ASSUMPTION_WEAKENING:
                            self._insert_in_table('assumptions modified', len(self.rule_builder.rules))
                        case Learning.GUARANTEE_WEAKENING:
                            self._insert_in_table('guarantees modified', len(self.rule_builder.rules))
                    self.rules_and_modifications.append(str(self.rule_builder))
            case RecordingState.SUMMARY:
                continues: bool = self.summary_builder.record(line)
                if not continues:
                    self._insert_in_table('choices', self.summary_builder.choices)
                    self._insert_in_table('solution status compared to ideal', self.summary_builder.solution_status)
                    if self.summary_builder.spec_id:
                        self._insert_in_table("id solution reached", self.summary_builder.spec_id)
                    elif "Success" in self.summary_builder.solution_status:
                        self._insert_in_table("id solution reached", 0)
                    elif self.cur_spec:
                        # TODO: add logic to record all solutions reached
                        id = self.spec_recorder.add(self.cur_spec)
                        self._insert_in_table("id solution reached", id)
                    else:
                        self._insert_in_table("id solution reached", 1)

                    self._insert_in_table('rules and modifications',
                                          "\n########\n".join(self.rules_and_modifications).strip())
                    self._insert_in_table('cleaned output (choices and results)',
                                          self.cleaned_output.strip())
                    self._reset_builder_for_new_row()
            case RecordingState.NOTHING:
                if re.search(r'Select an option by choosing its index:', line):
                    self.recording_state = RecordingState.CHOICES
                    self.choices_builder.start()
                elif re.search(r'Rule:', line):
                    self.recording_state = RecordingState.RULE_CHANGE
                    self.rule_builder.start()
                elif line.strip() == '#########################':
                    self.recording_state = RecordingState.SUMMARY
                    self.summary_builder.start()
                elif re.search(r'Moving to Guarantee Weakening', line):
                    self.phase = Learning.GUARANTEE_WEAKENING

        return False

    def _insert_in_table(self, col_name: str, value: Any):
        self._df[col_name].iloc[-1] = value

    def _increment_in_table(self, col_name: str):
        if not self._df[col_name].iloc[-1]:
            # Initialise int values with 0
            self._df[col_name].iloc[-1] = 0

        self._df[col_name].iloc[-1] += 1

    def save_to_file(self, path: str) -> str:
        if not os.path.exists(path):
            # TODO: BUG (file doesn't exist, but should be generated with given name)
            path = generate_temp_filename(ext=".csv")
            print(f"Writing csv at path: {path}")
        df = self.get_dataframe()
        df.to_csv(path, index=False)
        return path

    def get_dataframe(self):
        return self._df.dropna(how="all")

    def assert_line_for_errors(self, line):
        if re.match(r"Error:", line):
            raise ValueError(f"ERROR ENCOUNTERED DURING PARSING!!!:\n{line}")
