import argparse
from typing import Dict, Optional, Tuple
import os

from shield.bfs_repair_orchestrator import BFSRepairOrchestrator, SpecLogger
from spec_repair.components.interfaces.ilearner import ILearner
from spec_repair.components.optimising_final_spec_learner import OptimisingSpecLearner
from spec_repair.components.new_spec_oracle import NewSpecOracle
from spec_repair.components.learning_type_spec_mitigator import LearningTypeSpecMitigator
from spec_repair.components.spectra_discriminator import SpectraDiscriminator
from spec_repair.enums import Learning
from spec_repair.helpers.heuristic_managers.choose_first_heuristic_manager import ChooseFirstHeuristicManager
from spec_repair.helpers.recorders.unique_spec_recorder import UniqueSpecRecorder
from spec_repair.helpers.spectra_specification import SpectraSpecification
from spec_repair.util.file_util import write_to_file, read_file, read_file_lines
from spec_repair.config import PROJECT_PATH

from typing import List, Tuple, Dict

from spec_repair.util.mittigation_strategies import move_one_to_guarantee_weakening, complete_counter_traces


def run_single_repair(spec_path: str, trace_path: str, out_spec_path, out_test_dir_name: Optional[str] = None):
    spec: SpectraSpecification = SpectraSpecification.from_file(spec_path)
    trace: list[str] = read_file_lines(trace_path)
    learners: Dict[str, ILearner] = {
        "assumption_weakening": OptimisingSpecLearner(
            heuristic_manager=ChooseFirstHeuristicManager()
        ),
        "guarantee_weakening": OptimisingSpecLearner(
            heuristic_manager=ChooseFirstHeuristicManager()
        )
    }
    if out_test_dir_name:
        recorder = UniqueSpecRecorder(debug_folder=out_test_dir_name)
    else:
        recorder = UniqueSpecRecorder()
    repairer: BFSRepairOrchestrator = BFSRepairOrchestrator(
        learners,
        NewSpecOracle(),
        SpectraDiscriminator(),
        LearningTypeSpecMitigator({
            Learning.ASSUMPTION_WEAKENING: move_one_to_guarantee_weakening,
            Learning.GUARANTEE_WEAKENING: complete_counter_traces
        }),
        ChooseFirstHeuristicManager(),
        recorder,
        SpecLogger()
    )
    # Getting all possible repairs
    repairer.repair_bfs(spec, (trace, [], Learning.ASSUMPTION_WEAKENING, [], 0, 0))
    new_spec_strings: list[str] = [spec.to_str() for spec in recorder.get_all_values()]
    assert len(new_spec_strings) == 1, "Expected exactly one new specification after single repair."
    write_to_file(out_spec_path, new_spec_strings[0])
    return new_spec_strings


def main():
    parser = argparse.ArgumentParser(description='Run single repair on specification')
    parser.add_argument('spec_path', help='Path to the specification file')
    parser.add_argument('trace', help='Path to the trace file')
    parser.add_argument('out_spec_path', help='Path where to save the repaired specification')
    args = parser.parse_args()

    run_single_repair(args.spec_path, args.trace, args.out_spec_path)


if __name__ == "__main__":
    #run_single_repair(
    #    os.path.join(PROJECT_PATH, "tests/shield_test/specs/spec_0.spectra"),
    #    os.path.join(PROJECT_PATH, "tests/shield_test/specs/violation_trace_0.txt"),
    #    os.path.join(PROJECT_PATH, "tests/shield_test/specs/spec_1.spectra"),
    #)
    main()
