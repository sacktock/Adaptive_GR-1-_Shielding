import os


def _path_from_env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value:
        return os.path.expanduser(value)
    return os.path.expanduser(default)


PATH_TO_CLI = _path_from_env("GR1_SPECTRA_CLI", "~/SpectraTools/spectra-cli.jar")
PATH_TO_TOOLBOX = _path_from_env("GR1_SPECTRA_TOOLBOX", "~/SpectraTools/spectra_toolbox.jar")
PATH_TO_EXECUTOR = _path_from_env("GR1_SPECTRA_EXECUTOR", "~/SpectraTools/spectra-executor.jar")
PATH_TO_JVM = _path_from_env(
    "GR1_JVM_PATH",
    "/usr/lib/jvm/temurin-23-jdk-amd64/lib/server/libjvm.so",
)
PATH_TO_ILASP = _path_from_env("GR1_ILASP", "~/bin/ILASP")
PATH_TO_FASTLAS = _path_from_env("GR1_FASTLAS", "~/bin/FastLAS")
PRINT_CS = False
FASTLAS = False  # TODO: modify into enum (inductive ASP tool)
RESTORE_FIRST_HYPOTHESIS = True

# This determines the paths for running clingo and ILASP and whether to use
# Windows Subsystem for Linux (WSL):
SETUP_DICT = {'wsl': False,
              'clingo': 'clingo',
              'ILASP': PATH_TO_ILASP,
              'FastLAS': PATH_TO_FASTLAS,
              'ltlfilt': 'ltlfilt',
              'java': 'java',
              }

PROJECT_PATH: str = _path_from_env(
    "GR1_PROJECT_PATH",
    "/mnt/c/Users/alexw/OneDrive/Imperial/Projects/Adaptive-GR(1)-Shielding",
)
GENERATE_MULTIPLE_TRACES = False

# TODO: add these in a config class
MAX_ASP_HYPOTHESES = 10

# For testing and statistics
STATISTICS: bool = True
MANUAL: bool = True

# Configuration of Learning
WEAKENING_TO_JUSTICE = False
