import os
import threading
from typing import Optional, Dict

import jpype
import jpype.imports
import atexit
from jpype.types import *

from spec_repair.config import PATH_TO_JVM, PATH_TO_TOOLBOX, PATH_TO_CLI, PATH_TO_EXECUTOR

if not jpype.isJVMStarted():
    jpype.startJVM(PATH_TO_JVM, "-ea", classpath=[f"{PATH_TO_EXECUTOR}"])
    print("JVM started successfully")

AdaptiveShield = jpype.JClass('uk.ac.imperial.logix.AdaptiveShield')

def shutdown():
    def force_exit():
        print("Shutdown taking too long, forcing exit.")
        os._exit(1)

    print("Shutting down JVM...")
    timer = threading.Timer(10, force_exit)
    timer.start()

    # SpectraToolbox.shutdownNow()
    jpype.shutdownJVM()

    print("JVM shutdown initiated...")
    timer.cancel()
    print("JVM shutdown complete.")


atexit.register(shutdown)

def dict_to_java_hashmap(py_dict):
    HashMap = jpype.JClass('java.util.HashMap')
    jmap = HashMap()
    for k, v in py_dict.items():
        jmap.put(str(k), str(v))  # convert keys/values to strings explicitly
    return jmap

def java_hashmap_to_dict(safe_output_java) -> Dict[str, str]:
    # Convert Java Map<String, String> back to Python dict
    py_safe_output = {}
    iterator = safe_output_java.entrySet().iterator()
    while iterator.hasNext():
        entry = iterator.next()
        py_safe_output[str(entry.getKey())] = str(entry.getValue())
    return py_safe_output

class ControllerShield:
    def __init__(self, folder_path: str):
        """
        Initialize ControllerShield by connecting to the Java server and
        sending the folder path for initialization.
        """
        self._shield = AdaptiveShield(folder_path)

    def initiate_starting_state(self, state: Optional[Dict[str,str]] = None) -> bool:
        if state is None:
            state = {}
        current_state = dict_to_java_hashmap(state)
        return self._shield.initialiseStartingState(current_state)

    def get_safe_action(self, state, action) -> Dict[str, str]:
        current_inputs = dict_to_java_hashmap(state)
        current_outputs = dict_to_java_hashmap(action)

        safe_output_java = self._shield.getSafeAction(current_inputs, current_outputs)
        safe_action = java_hashmap_to_dict(safe_output_java)
        return safe_action