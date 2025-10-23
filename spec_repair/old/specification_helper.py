import subprocess
import os
import re

from spec_repair.config import SETUP_DICT


def print_dict(dict):
    for key in dict.keys():
        print(key + "\t:\t" + dict.get(key))


def dict_to_text(dict):
    output = ""
    for key in dict.keys():
        output += key + "\t:\t" + dict.get(key) + "\n"
    return output


def get_folders(folder, exclusions=[]):
    files = [os.path.join(folder, x) for x in os.listdir(folder) if x not in exclusions]
    folders = [x for x in files if os.path.isdir(x)]
    return folders


# TODO: Fix infinite recursion
def run_subprocess(cmd, encoding: str = 'utf-8', suppress=False, timeout=-1):
    # timed = timeout > 0
    # print("Running command: ", " ".join(cmd))
    if suppress:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)#, start_new_session=timed)
    else:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)#, start_new_session=timed)
    # if timed:
    #     try:
    #         p.wait(timeout=timeout)
    #     except subprocess.TimeoutExpired:
    #         print("Subprocess Timeout")
    #         os.kill(p.pid, signal.CTRL_C_EVENT)
    #         subprocess.call(['taskkill', '/F', '/T', '/PID', str(p.pid)])
    #         return "Timeout"
    output = p.communicate()[0]
    output = output.decode(encoding)
    return output


def assign_equalities(formula_n, variables):
    for var in variables:
        formula_n = re.sub("!" + var + "(?!=|[a-z])", var + "=false", formula_n)
        formula_n = re.sub(var + "(?!=|[a-z])", var + "=true", formula_n)
    return formula_n


def strip_vars(spec, sub=["env", "sys"]):
    return re.findall(r"[" + '|'.join(sub) + r"]\s*boolean\s*(.*)\s*;", '\n'.join(spec))
    variables = []
    for line in spec:
        words = line.split(" ")
        if words[0] in sub and words[0] != "":
            search = re.compile("boolean\s*(.*)\s*;").search(line)
            if search:
                var = re.sub("\s", "", search.group(1))
                variables.append(var)
    return variables


def get_name(filename):
    path = filename.split("/")
    name = path[len(path) - 2]
    name = re.sub(r"\s", r"_", name)
    return name.title()


CASE_STUDY_EXCLUSION_LIST = ['acheivepattern',
                             'atm',
                             'detector',
                             'lily01',
                             'lily02',
                             'lily11',
                             'lily15',
                             'lily16',
                             'ltl2dba_R_2',
                             'ltl2dba_theta_2',
                             'ltl2dba27',
                             'prioritizedArbiter',
                             'retractionPattern2',
                             'tcp',
                             'telephone']

CASE_STUDY_FINALS = {  # "../Translators/input-files/examples/Arbiter/Arbiter_FINAL.spectra",
        "Lift": "../Translators/input-files/examples/lift_FINAL.spectra",
        "Lift New": "../Translators/input-files/examples/lift_FINAL_NEW.spectra",
        "Minepump": "../Translators/input-files/case-studies/modified-specs/minepump/genuine/minepump_FINAL.spectra",
        "Traffic Single": "../Translators/input-files/examples/Traffic/traffic_single_FINAL.spectra",
        "Traffic": "../Translators/input-files/examples/Traffic/traffic_updated_FINAL.spectra"}


def create_cmd(param):
    cmd = []
    if SETUP_DICT['wsl']:
        cmd.append('wsl')
    cmd.append(SETUP_DICT[param[0]])
    if len(param) == 1:
        return cmd
    cmd += param[1:]
    return cmd
