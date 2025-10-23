import copy
import re
from typing import Optional

from spec_repair.config import PATH_TO_CLI
from spec_repair.ltl_types import CounterStrategy
from spec_repair.old.patterns import PRS_REG
from spec_repair.old.specification_helper import run_subprocess
from spec_repair.util.file_util import generate_temp_filename, write_to_file
from spec_repair.util.spec_util import synthesise_extract_counter_strategies


class SpecOracle:
    def __init__(self):
        self._response_pattern = """\
pattern pRespondsToS(s, p) {
  var { S0, S1} state;

  // initial assignments: initial state
  ini state=S0;

  // safety this and next state
  alw ((state=S0 & ((!s) | (s & p)) & next(state=S0)) |
  (state=S0 & (s & !p) & next(state=S1)) |
  (state=S1 & (p) & next(state=S0)) |
  (state=S1 & (!p) & next(state=S1)));

  // equivalence of satisfaction
  alwEv (state=S0);
}"""

    def synthesise_and_check(self, spec: list[str]) -> Optional[CounterStrategy]:
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

    def _synthesise(self, spec):
        spec = self._pRespondsToS_substitution(spec)
        spectra_file: str = generate_temp_filename(ext=".spectra")
        write_to_file(spectra_file, '\n'.join(spec))
        return synthesise_extract_counter_strategies(spectra_file)

    def _pRespondsToS_substitution(self, spec: list[str]) -> list[str]:
        spec = copy.deepcopy(spec)
        is_necessary = False
        for i, line in enumerate(spec):
            line = line.strip("\t|\n|;")
            if PRS_REG.search(line):
                is_necessary = True
                s = re.search(r"G\(([^-]*)", line).group(1)
                p_matches = re.findall(r"F\((.*?)\)", line)
                if p_matches:
                    p = "|".join(p_matches)
                else:
                    raise ValueError(f"Trouble extracting p from: {line}")
                spec[i] = f"\tpRespondsToS({s},{p});\n"
        if is_necessary:
            spec.append(self._response_pattern)
        return spec
