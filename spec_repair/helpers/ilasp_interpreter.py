import re
from typing import Optional, List, Tuple

from spec_repair.exceptions import NoViolationException
from spec_repair.helpers.adaptation_learned import Adaptation


class ILASPInterpreter:

    @staticmethod
    def extract_learned_possible_adaptations_raw(output: str) -> Optional[List[Tuple[int, List[str]]]]:
        if re.search("UNSATISFIABLE", ''.join(output)):
            return None
        if re.search(r"1 \(score 0\)", ''.join(output)):
            raise NoViolationException("Learning problem is trivially solvable. "
                                       "If spec is not realisable, we have a learning error.")

        pattern = re.compile(r"%% Solution \d+ \(score (\d+)\)\s+((?:[^%]+(?:\n|$))*)")
        matches = pattern.findall(output)

        # Split each match into lines and remove empty lines, returning as (score, list of lines)
        return [(int(score), list(filter(None, match.split('\n')))) for score, match in matches]

    @staticmethod
    def extract_learned_possible_adaptations(output: str) -> Optional[List[Tuple[int, List[Adaptation]]]]:
        possible_adaptations: Optional[List[List[str]]] = ILASPInterpreter.extract_learned_possible_adaptations_raw(output)
        if not possible_adaptations:
            return None
        return [(score, [Adaptation.from_str(adaptation) for adaptation in adaptations])
                for score, adaptations in possible_adaptations]
