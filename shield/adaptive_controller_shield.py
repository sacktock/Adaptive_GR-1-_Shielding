import os
import shutil
from typing import Optional, Dict, List, Tuple

import subprocess
import sys

from shield.controller_shield import ControllerShield
from spec_repair.config import PATH_TO_CLI, PROJECT_PATH
from spec_repair.old.specification_helper import run_subprocess
from spec_repair.util.file_util import write_to_file


def synthesise_controller(spec_file_path, output_folder_path):
    cmd = ['java', '-jar', PATH_TO_CLI, '-i', spec_file_path, '--jtlv', '-s', '--static', '-o', output_folder_path]

    return run_subprocess(cmd)


def generate_holds_at_statements(
        trace: List[Tuple[Dict[str, str], Dict[str, str]]],
        trace_name: str = "trace_name_0"
) -> List[str]:
    statements = []
    for idx, (state_dict, action_dict) in enumerate(trace):
        for key, value in state_dict.items():
            prefix = "" if value.lower() == "true" else "not_"
            statements.append(f"{prefix}holds_at({key},{idx},{trace_name}).")
        for key, value in action_dict.items():
            prefix = "" if value.lower() == "true" else "not_"
            statements.append(f"{prefix}holds_at({key},{idx},{trace_name}).")
    return statements


def run_jvm_module_subprocess(script, input_file, config_file, output_file):
    print(f"Running {script}")
    print(f"Input file: {input_file}")
    print(f"Config file: {config_file}")
    print(f"Output file: {output_file}")
    cmd = [
        sys.executable,
        "-m", script,
        input_file, config_file, output_file
    ]
    process = subprocess.run(
        cmd,
        cwd=str(PROJECT_PATH),
        capture_output=True,
        text=True,
        check=False,
    )

    print("=== Subprocess STDOUT ===")
    print(process.stdout)
    print("=== Subprocess STDERR ===")
    print(process.stderr)

    if process.returncode != 0:
        raise RuntimeError(f"Subprocess failed with exit code {process.returncode}")


class AdaptiveShield:
    def __init__(self, spec_path: str, index: int, initiate_synthesis: bool = False, initiate_spec_repair: bool = True):
        """
        Initialize AdaptiveControllerShield by connecting to the Java server
        and sending the folder path for initialization.
        """
        if not spec_path.endswith('.spectra'):
            raise ValueError("Specification file must have .spectra extension")
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Specification file not found at path: {spec_path}")

        self.initiate_synthesis = initiate_synthesis
        self.initiate_spec_repair = initiate_spec_repair

        parent_folder_path = os.path.dirname(spec_path)
        self._id = 0
        self._index = index
        # Create storage folders for specifications and controllers
        self._specs_folder_path = os.path.join(parent_folder_path, f'specs_{self._index}')
        os.makedirs(self._specs_folder_path, exist_ok=True)
        self._controllers_folder_path = os.path.join(parent_folder_path, f'controllers_{self._index}')
        os.makedirs(self._controllers_folder_path, exist_ok=True)

        # Copy the first specification file to the specs folder
        first_spec_path = self._get_spec_path()
        shutil.copy2(spec_path, first_spec_path)

        # Generate the first controller
        first_controller_folder_path = self._get_controller_folder_path()
        if self.initiate_synthesis:
            res = synthesise_controller(first_spec_path, first_controller_folder_path)
            if res:
                print(res)

        # Initialize the ControllerShield with the first controller folder path
        self._shield = ControllerShield(first_controller_folder_path)

        self._trace_log = []

    def initiate_starting_state(self, state: Optional[Dict[str, str]] = None) -> bool:
        if state is None:
            state = {}
        print(state)
        return self._shield.initiate_starting_state(state)

    def get_safe_action(self, state, action):
        """
        Get a safe action based on the current state and proposed action.
        This method overrides the parent class method to provide adaptive behavior.
        """
        safe_action: Dict[str, str] = self._shield.get_safe_action(state, action)
        if not safe_action:
            """
            Optional skipping of shield repair subprocess, since we've already repaired the spec previously
            """
            if self.initiate_spec_repair:
                run_jvm_module_subprocess("shield.repair_specification",
                                          self._get_spec_path(), self._get_violation_trace_file(),
                                          self._get_next_spec_path())
            new_controller_folder_path = self._get_next_controller_folder_path()
            # raise RuntimeError("Should not need to repair")
            if self.initiate_spec_repair:
                self._id += 1
            if self.initiate_synthesis:
                res = synthesise_controller(self._get_next_spec_path(), new_controller_folder_path)
                if res:
                    print(res)
            self._shield = ControllerShield(new_controller_folder_path)
            self._update_shield_with_trace()
            safe_action: Dict[str, str] = self._shield.get_safe_action(state, action)
            print("Safe action after spec repair:", safe_action)
            
        self._trace_log.append((state, {k: v for k, v in safe_action.items() if k in action}))
        return safe_action

    def cleanup(self):
        shutil.rmtree(self._specs_folder_path, ignore_errors=False, onerror=None)
        shutil.rmtree(self._controllers_folder_path, ignore_errors=False, onerror=None)

    def _get_violation_trace_file(self):
        violation_trace = generate_holds_at_statements(self._trace_log)
        trace_file_path = os.path.join(self._specs_folder_path, f'violation_trace_{self._id}.txt')
        write_to_file(trace_file_path, "\n".join(violation_trace))

        return trace_file_path

    def _get_spec_path(self):
        return os.path.join(self._specs_folder_path, f'spec_{self._id}.spectra')

    def _get_next_spec_path(self):
        return os.path.join(self._specs_folder_path, f'spec_{self._id + 1}.spectra')

    def _get_controller_folder_path(self):
        return os.path.join(self._controllers_folder_path, f'controller_{self._id}')

    def _get_next_controller_folder_path(self):
        return os.path.join(self._controllers_folder_path, f'controller_{self._id + 1}')

    def _update_shield_with_trace(self):
        self._shield.initiate_starting_state()
        for i, (state, action) in enumerate(self._trace_log):
            print(f"[{i}]")
            safe_action = self._shield.get_safe_action(state, action)
            assert {k: v for k, v in safe_action.items() if k in action} == action, f"Safe action {safe_action} at timepoint {i} with state {state} does not match proposed action {action}"


PATH_TO_SPEC = os.path.join(PROJECT_PATH, "tests/shield_test/minepump_strong.spectra")

if __name__ == "__main__":
    print(PROJECT_PATH)
    shield = AdaptiveShield(spec_path=PATH_TO_SPEC)
    state_minus1 = {"methane": "false", "highwater": "false"}
    shield.initiate_starting_state(state_minus1)

    # Zeroth test case
    print("Case [0]")
    state0 = {"methane": "false", "highwater": "false"}
    action0 = {"pump": "false"}
    safe_output0 = shield.get_safe_action(state0, action0)
    print(safe_output0)

    # First test case
    print("Case [1]")
    state1 = {"methane": "false", "highwater": "true"}
    action1 = {"pump": "false"}
    safe_output1 = shield.get_safe_action(state1, action1)
    print(safe_output1)

    # Second test case
    print("Case [2]")
    state2 = {"methane": "false", "highwater": "true"}
    action2 = {"pump": "false"}
    safe_output2 = shield.get_safe_action(state2, action2)
    print(safe_output2)

    # Third test case
    print("Case [3]")
    state3 = {"methane": "true", "highwater": "true"}
    action3 = {"pump": "true"}
    safe_output3 = shield.get_safe_action(state3, action3)
    print(safe_output3)

    # Fourth test case
    print("Case [4]")
    state4 = {"methane": "false", "highwater": "false"}
    action4 = {"pump": "false"}
    safe_output4 = shield.get_safe_action(state4, action4)
    print(safe_output4)
