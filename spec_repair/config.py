import os.path

PATH_TO_CLI = os.path.expanduser("~/SpectraTools/spectra-cli.jar")
PATH_TO_CORES = os.path.expanduser("~/SpectraTools/spectra_unrealizable_cores.jar")
PATH_TO_ALL_CORES = os.path.expanduser("~Spectra/Tools/spectra_all_unrealisable_cores.jar")
PATH_TO_TOOLBOX = os.path.expanduser("~/SpectraTools/spectra_toolbox.jar")
PATH_TO_EXECUTOR = os.path.expanduser("~/SpectraTools/spectra-executor.jar")
PATH_TO_JVM = "/usr/lib/jvm/temurin-23-jdk-amd64/lib/server/libjvm.so"
PATH_TO_ILASP = os.path.expanduser('~/bin/ILASP')
PATH_TO_FASTLAS = os.path.expanduser('~/bin/FastLAS')
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

# TODO - update this to your use home directory
USER_PATH = 'path_to_user'

# TODO - update this to your project path
PROJECT_PATH: str = os.path.expanduser("path_to_project")
GENERATE_MULTIPLE_TRACES = False

# Violation Listening Configurations
LOG_FOLDER = f'{USER_PATH}/eclipse-workspace/logs'

# TODO: add these in a config class
MAX_ASP_HYPOTHESES = 10

# For testing and statistics
STATISTICS: bool = True
MANUAL: bool = True

# Configuration of Learning
WEAKENING_TO_JUSTICE = False
