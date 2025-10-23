from spec_repair.components.repair_orchestrator import RepairOrchestrator
from spec_repair.components.spec_learner import SpecLearner
from spec_repair.components.spec_oracle import SpecOracle
from spec_repair.config import PROJECT_PATH
from spec_repair.ltl_types import Trace
from spec_repair.util.file_util import generate_temp_filename
from spec_repair.util.spec_util import get_assumptions_and_guarantees_from, generate_trace_asp


# TODO: use this file as replacement for Specification.py logic


def generate_asp_trace_to_file(start_file: str, end_file: str) -> str:
    trace_file = generate_temp_filename(".txt")
    generate_trace_asp(start_file, end_file, trace_file)
    return trace_file


if __name__ == '__main__':
    start_file = f"{PROJECT_PATH}/input-files/examples/Minepump/minepump_strong.spectra"
    end_file = f"{PROJECT_PATH}/input-files/examples/Minepump/minepump_FINAL.spectra"
    trace_file = generate_asp_trace_to_file(start_file, end_file)
    spec_df = get_assumptions_and_guarantees_from(start_file)
    trace = Trace(trace_file)
    spec_repairer = RepairOrchestrator(learner=SpecLearner(), oracle=SpecOracle())
    spec_repairer.repair_spec(spec_df, trace)
