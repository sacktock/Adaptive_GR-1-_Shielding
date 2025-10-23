import re
from copy import deepcopy
from typing import Optional, List, Tuple

from spec_repair.components.interfaces.ioracle import IOracle
from spec_repair.components.spec_oracle import SpecOracle
from spec_repair.config import PATH_TO_CLI
from spec_repair.enums import Learning
from spec_repair.helpers.counter_trace import cts_from_cs, CounterTrace
from spec_repair.helpers.spectra_specification import SpectraSpecification
from spec_repair.ltl_types import CounterStrategy
from spec_repair.old.specification_helper import run_subprocess
from spec_repair.util.file_util import generate_temp_filename, write_to_file
from spec_repair.util.spec_util import synthesise_extract_counter_strategies


class NewSpecOracle(IOracle):
    def __init__(self):
        self._ct_cnt = 0

    def is_valid_or_counter_arguments(
            self,
            new_spec: SpectraSpecification,
            data: Tuple[list[str], list[CounterTrace], Learning, list[SpectraSpecification], int, float]
    ) -> Optional[List[Tuple[CounterTrace, Tuple[list[str], list[CounterTrace], Learning, list[SpectraSpecification], int, float]]]]:
        counter_strategy = self._synthesise_and_check(new_spec)
        if counter_strategy:
            possible_counter_traces = cts_from_cs(counter_strategy, cs_id=self._ct_cnt)
            if self._hm:
                possible_counter_traces = self._hm.select_counter_traces(possible_counter_traces)
            self._ct_cnt += 1
            possible_counter_traces_with_data = [(possible_counter_trace, deepcopy(data)) for possible_counter_trace in possible_counter_traces]
            return possible_counter_traces_with_data
        else:
            return None

    def _synthesise_and_check(self, spec: SpectraSpecification) -> Optional[CounterStrategy]:
        """
        Uses Spectra under the hood to check whether specifcation is realisable.
        If it is, nothing is returned. Otherwise, it returns a CounterStrategy.
        """
        output = self._synthesise(spec)
        if re.search("Result: Specification is unrealizable", output):
            output = str(output).split("\n")
            counter_strategy = list(filter(re.compile(r"\s*->\s*[^{]*{[^}]*").search, output))
            return counter_strategy
        elif re.search("Result: Specification is realizable", output):
            return None
        else:
            raise Exception(output)

    def _synthesise(self, spec: SpectraSpecification):
        spec_str = spec.to_str(is_to_compile=True)
        spectra_file: str = generate_temp_filename(ext=".spectra")
        write_to_file(spectra_file, spec_str)
        return synthesise_extract_counter_strategies(spectra_file)
