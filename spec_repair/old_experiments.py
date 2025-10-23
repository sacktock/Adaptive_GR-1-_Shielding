import os
import random
import re
import shutil
import subprocess
import time
from collections import deque
from datetime import datetime as dt
from itertools import product
from statistics import median
from typing import Optional, List

import numpy as np
import pandas as pd
import sympy
from anytree import AnyNode
from matplotlib import pyplot as plt
from sympy import parse_expr

from spec_repair.old.patterns import PRS_REG
from spec_repair.util.spec_util import format_spec, extract_df_content, generate_trace_asp, write_trace, \
    extract_expressions_from_file, generate_model, simplify_assignments, extract_all_expressions, run_clingo_raw, \
    semantically_identical_spot, extract_all_expressions_spot, realizable, negate
from spec_repair.old.case_study_translator import delete_files, parenthetic_contents_with_function, translate_case_study, negate_and_simplify
from spec_repair.config import PROJECT_PATH, FASTLAS, GENERATE_MULTIPLE_TRACES, PATH_TO_CLI, PRINT_CS
from spec_repair.enums import SimEnv, Outcome, Learning, When
from spec_repair.old.latex_translator import spectra_to_latex, violation_to_latex
from spec_repair.old.specification_helper import strip_vars, get_folders, \
    CASE_STUDY_EXCLUSION_LIST, \
    dict_to_text, print_dict, CASE_STUDY_FINALS, run_subprocess
from spec_repair.util.file_util import generate_filename, generate_random_string, read_file_lines, write_file, \
    generate_temp_filename


def traffic_weakening_modified(include_prev, unguided_learning=False):
    spectra_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_updated.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_violation_long.txt"
    violation_list = ['carA_idle_when_red']
    if unguided_learning:
        violation_list = []

    self = Specification(spectra_file, violation_file, violation_list, include_prev)
    self.run_pipeline()


def traffic_weakening_example(include_prev, unguided_learning=False):
    spectra_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_violation.txt"
    violation_list = ['carA_idle_when_red']
    if unguided_learning:
        violation_list = []

    self = Specification(spectra_file, violation_file, violation_list, include_prev)
    self.run_pipeline()


def traffic_single():
    spectra_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_violation.txt"
    violation_list = ['car_moves_when_green']

    self = Specification(spectra_file, violation_file, violation_list, include_prev=False)
    self.run_pipeline()


def traffic_weaken_to_unrealizable_example(include_prev, unguided_learning=False):
    spectra_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_for_strengthen_A_only.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_for_strengthen_violation_single_A_only.txt"
    violation_list = ['carA_moves_when_green']
    if unguided_learning:
        violation_list = []

    traffic_A_only = Specification(spectra_file, violation_file, violation_list, include_prev)
    traffic_A_only.run_pipeline()


def traffic_guarantee_weaken_single_simplified():
    spectra_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_guarantee_simple.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_guarantee_violation_simple.txt"
    violation_list = ['car_moves_when_green']

    self = Specification(spectra_file, violation_file, violation_list, include_prev=False)
    self.run_pipeline()

    # run_clingo(spectra_file)


def ForkLiftDelay():
    spectra_file = f"{PROJECT_PATH}/input-files/examples/ForkliftDelay.spectra"
    self = Specification(spectra_file, "", [], include_prev=False)
    self.format_spectra_file()
    self.generate_formula_df_from_spectra()


def minepump():
    spectra_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed0_modified.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed0_violation_modified.txt"
    violation_list = ["strong_assumption"]
    spectra_to_latex(spectra_file)
    violation_to_latex(violation_file)
    self = Specification(spectra_file, violation_file, violation_list, include_prev=False, include_next=True)
    # self.format_spectra_file()
    # self.spectra_to_DataFrames()

    self.run_pipeline()


def strengthen_all():
    files = {
        "Minepump":
            {
                "start": f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed0.spectra",
                "end": f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed1.spectra"
            },
        # "Arbiter":
        #     {"start": f"{PROJECT_PATH}/input-files/examples/Arbiter/Arbiter_FINAL_dropped0.spectra",
        #      "end": f"{PROJECT_PATH}/input-files/examples/Arbiter/Arbiter_FINAL.spectra"
        #      },
        # "Lift":
        #     {  # "start": f"{PROJECT_PATH}/input-files/examples/lift_FINAL_NEW_strong.spectra",
        #         # "end": f"{PROJECT_PATH}/input-files/examples/lift_FINAL_NEW.spectra"
        #         "start": f"{PROJECT_PATH}/input-files/examples/lift_well_sep_strong.spectra",
        #         "end": f"{PROJECT_PATH}/input-files/examples/lift_well_sep.spectra"
        #     },
        # "Traffic Single":
        #     {"start": f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_FINAL_strong.spectra",
        #      "end": f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_FINAL.spectra"
        #      },
        # "Traffic":
        #     {"start": f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_updated_FINAL_strong.spectra",
        #      "end": f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_updated_FINAL.spectra"
        #      },
        # "Genbuf":
        #     {"start": f"{PROJECT_PATH}/input-files/examples/genbuf_05_normalised_dropped.spectra",
        #      "end": f"{PROJECT_PATH}/input-files/examples/genbuf_05_normalised.spectra"
        #      }
    }

    for include_prev in [True]:
        for name in files.keys():
            strengthen_n(files[name]["end"], include_prev)


def strengthen_n(spectra_file, include_prev):
    folder = "non_temporal"
    if include_prev:
        folder = "temporal"
    name = re.search(r"/([^/]*)\.spectra", spectra_file).group(1)
    expressions = ["assumption", "guarantee"]
    end_file = format_iff(spectra_file)
    old_specs = [''.join(read_file_lines(end_file))]
    start_files = []
    repeats = 0
    count = 0
    while True:
        start_file, n, spec = drop_random(end_file, expressions, 0, include_prev, write=False)
        if spec in old_specs:
            repeats += 1
            if repeats > 20 or count > 10:
                break
            continue
        if start_file == "none exist":
            print("none exist")
            return None
        start_file = f"{PROJECT_PATH}/input-files/strengthened/{folder}/{name}_dropped{count}.spectra"
        write_file(start_file, spec)
        if realizable(start_file, suppress=True):
            # if not contains_contradictions(start_file, "assumption|asm"):
            old_specs.append(spec)
            start_files.append(start_file)
            count += 1
    # return start_files


def genbuf():
    spectra_file = f"{PROJECT_PATH}/input-files/examples/genbuf_05.spectra"
    results = drop_and_evaluate(spectra_file, limit=1)


def genbuf_simple():
    # simple:
    # spectra_file = f"{PROJECT_PATH}/input-files/examples/genbuf_05_normalised_dropped.spectra"
    # violation_file = f"{PROJECT_PATH}/input-files/examples/genbuf_05_normalised_dropped_auto_violation.txt"

    # not so simple:
    spectra_file = f"{PROJECT_PATH}/input-files/strengthened/temporal/genbuf_05_normalised_dropped10.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/strengthened/temporal/genbuf_05_normalised_dropped10_auto_violation_temp.txt"
    self = Specification(spectra_file, violation_file)
    self.run_pipeline(timeout_value=6000)


def minepump_auto():
    spectra_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed1.spectra"
    exp = ["assumption"]
    results = drop_and_evaluate(spectra_file, expressions=exp)
    # df = results_to_df(results, spectra_file)


def minepump_trial():
    start_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed1_normalised_dropped.spectra"
    trace_file = generate_filename(start_file, "_auto_violation.txt")
    self = Specification(start_file, trace_file, violation_list=[], include_prev=False)
    elapsed = self.run_pipeline()
    self.violation_list


def trial_specs(end_file, start_files):
    # TODO: use these to figure out why some specs end unrealizable. - believe to do with deadlocks
    # end_file = f"{PROJECT_PATH}/input-files/examples/lift_FINAL.spectra"
    end_file = format_iff(end_file)

    # start_files = ["output-files/dropped/lift_FINAL_normalised_dropped114.spectra",
    #                "output-files/dropped/lift_FINAL_normalised_dropped113.spectra",
    #                "output-files/dropped/lift_FINAL_normalised_dropped107.spectra"]
    results = {}
    for i, start_file in enumerate(start_files):
        outcome, learning_time = run_simulated_environment_jit(start_file, end_file, limit=10, include_prev=False)
        outcome.print()
        results[i] = (outcome, learning_time, 0, [])
    df = results_to_df(results, end_file, types=["A,G"], include_prev=False)
    save_results_df(df)


def drop_and_evaluate(spectra_file, repeat=1, limit=10, expressions=["assumption"], include_prev=False):
    end_file = format_iff(spectra_file)
    # print(realizable(new_file))
    # print(realizable(spectra_file))
    results = {}
    old_specs = [''.join(read_file_lines(spectra_file))]
    repeats = 0
    for i in range(repeat):
        while True:
            start_file, n, spec = drop_random(end_file, expressions, 0, include_prev)
            if spec in old_specs:
                repeats += 1
                if repeats > 10:
                    break
                continue
            if start_file == "none exist":
                print("none exist")
                return None
            if realizable(start_file, suppress=True):
                old_specs.append(spec)
                break
            # while not realizable(start_file, suppress=True):
            #     start_file, n = drop_random(end_file, expressions, 0)
        if repeats > 10:
            continue
        outcome, learning_time = run_simulated_environment_jit(start_file, end_file, limit, include_prev)
        results[i] = (outcome, learning_time, n, old_specs)
        save_files_to_dropped(start_file)
    return results


def save_files_to_dropped(start_file):
    reg = re.search(r"^(.*)/([^/]*)\.spectra$", start_file)
    if not reg:
        return
    folder = "output-files/dropped"
    files = os.listdir(folder)
    name = reg.group(2)
    matches = re.findall(name + "(\d*)", ' '.join(files))
    if len(matches) == 0:
        n = 0
    else:
        n = max([int(x) for x in matches]) + 1
    shutil.copyfile(start_file, folder + "/" + name + str(n) + ".spectra")
    fixed_file = reg.group(1) + "/" + name + "_fixed.spectra"
    try:
        shutil.copyfile(fixed_file, folder + "/" + name + str(n) + "_fixed.spectra")
    except FileNotFoundError:
        return


def symplify(line, variables):
    # line = '\tG(car -> F(green));\n'
    if re.search(r"\bboolean\b|\bassumption\b|\bass\b|\bgar\b|\bguarantee\b|\bmodule\b|\bspec\b", line):
        return line
    line = re.sub(r"\s", "", line)
    if line == "":
        return line
    response = re.search(r'^\s*G\(([^-]*)->\s*F\((.*)\)\s*\)\s*;', line)
    if response:
        return "\t" + line + "\n"
    liveness = re.search(r"GF\s*\((.*)\)\s*;", line)
    justice = re.search(r"G\s*\((.*)\)\s*;", line)
    prefix = "\t("
    if liveness:
        prefix = "\tGF("
        line = liveness.group(1)
    if justice:
        prefix = "\tG("
        line = justice.group(1)
    if response:
        prefix = "\tG("
        cons = response.group(2)
        line = response.group(1) + "->" + re.sub(r"(" + "|".join(variables) + ")", r"\1_F", cons)
    line = re.sub(";", "", line)
    line = re.sub(r"next\(([^\)]*)\)", r"\1_X", line)
    line = re.sub(r"PREV\(([^\)]*)\)", r"\1_P", line)
    line = re.sub(r"F\(([^\)]*)\)", r"\1_F", line)
    line = re.sub("!", "~", line)
    line = re.sub("->", ">>", line)
    line = re.sub("<-", "<<", line)
    line = re.sub(r"^([^\(^\)]*)>>", r"(\1)>>", line)
    line = re.sub(r">>([^\(^\)]*)$", r">>(\1)", line)

    parsed = parse_expr(line)
    dnf = sympy.to_dnf(parsed)
    output = re.sub(r"~", "!", str(dnf))
    output = re.sub(r"(!?)\b([^\s]*)_X", r"next(\1\2)", output)
    output = re.sub(r"(!?)\b([^\s]*)_P", r"PREV(\1\2)", output)
    output = re.sub(r"(!?)\b([^\s]*)_F", r"F(\1\2)", output)
    return prefix + output + ");\n"


def name_expressions(spec):
    count = 0
    for i, line in enumerate(spec):
        exp_type = re.search(r"(assumption|guarantee)", line)
        if exp_type:
            if not re.search(r"(assumption|guarantee)\s*--\s*[a-zA-z]+", line):
                exp = exp_type.group(1)
                spec[i] = (exp + " -- " + exp + str(count) + '\n')
                count += 1
    return spec


def lower_variables(spec, variables):
    mapping = {var: var.lower() for var in variables}
    for key in mapping.keys():
        spec = [re.sub(r"\b" + re.escape(key) + r"\b", mapping[key], line) for line in spec]
    return spec, list(mapping.values())


def format_iff(spectra_file):
    out_file = generate_filename(spectra_file, "_normalised.spectra")
    spec = read_file_lines(spectra_file)
    variables = strip_vars(spec)
    spec, variables = lower_variables(spec, variables)
    spec = [re.sub(r"next\(([^\)]*)\)", r"(next(\1))", line) for line in spec]
    spec = format_spec(spec)
    spec = simplify_assignments(spec, variables)
    spec = name_expressions(spec)
    spec = [iff_to_dnf(line) for line in spec]
    spec = [symplify(line, variables) for line in spec]
    write_file(out_file, spec)
    return out_file


def trial_unrealizable():
    end_file = f"{PROJECT_PATH}/input-files/examples/lift_FINAL.spectra"

    start_files = ["output-files/dropped/lift_FINAL_normalised_dropped114.spectra",
                   "output-files/dropped/lift_FINAL_normalised_dropped113.spectra",
                   "output-files/dropped/lift_FINAL_normalised_dropped107.spectra"]

    start_files = ["output-files/dropped/lift_FINAL_normalised_dropped152.spectra"]
    start_files = ["output-files/dropped/lift_FINAL_normalised_dropped242.spectra"]

    # end_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_updated_FINAL.spectra"
    # start_files = ["output-files/dropped/traffic_updated_FINAL_normalised_dropped143.spectra"]
    trial_specs(end_file, start_files)


def genbuf_trial():
    spectra_file = f"{PROJECT_PATH}/input-files/examples/genbuf_05_normalised_dropped.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/examples/genbuf_05_normalised_dropped_auto_violation.txt"
    self = Specification(spectra_file, violation_file)
    self.format_spectra_file()
    self.generate_formula_df_from_spectra()


def minepump_motivating_example():
    # These are the ones with initial conditions:
    start_file = f"{PROJECT_PATH}/input-files/examples/Minepump/minepump_strong.spectra"
    end_file = f"{PROJECT_PATH}/input-files/examples/Minepump/minepump_FINAL.spectra"

    trace_file = generate_filename(start_file, "_auto_violation.txt")
    delete_files(trace_file)
    generate_trace_asp(start_file, end_file, trace_file)
    self = Specification(start_file, trace_file, include_next=False, random_hypothesis=True)
    self.run_pipeline()


def minepump_motivating_example_thread_safe():
    start_file = f"{PROJECT_PATH}/input-files/examples/Minepump/minepump_strong.spectra"
    end_file = f"{PROJECT_PATH}/input-files/examples/Minepump/minepump_FINAL.spectra"

    random_suffix = generate_random_string(length=10)
    trace_file = generate_filename(start_file, f"_auto_violation_{random_suffix}.txt")
    generate_trace_asp(start_file, end_file, trace_file)
    self = Specification(start_file, trace_file, include_next=False, random_hypothesis=True)
    self.run_pipeline()


def minepump_from_simulated_environment():
    start_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed0_modified.spectra"
    end_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed0.spectra"
    outcome, _ = run_simulated_environment(start_file, end_file)
    outcome.print()


def minepump2_from_simulated_environment():
    start_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed0.spectra"
    end_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed1.spectra"
    outcome, _ = run_simulated_environment(start_file, end_file)
    outcome.print()


def traffic_single_light_from_simulated_environment():
    start_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single.spectra"
    end_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_FINAL.spectra"
    run_simulated_environment(start_file, end_file)


def examples():
    arbiter_motivating_example()
    # minepump()
    # example_case()
    # lift()
    # minepump_from_simulated_environment()
    # traffic_single_light_from_simulated_environment()
    # minepump2_from_simulated_environment()
    # genbuf()
    # traffic_guarantee_weaken_single()
    # minepump_auto()
    # traffic_update_trial()

    # NB: drop_and_run is the large evaluation:
    # drop_and_run()
    # trial_unrealizable()
    # trial_lift_unrealizable_ass_only()
    # lift_91_no_trace_trial()
    minepump_motivating_example()
    # trial_traffic_48_43()
    # lift_trial()
    # traffic_guarantee_weaken_single()


def lift_motivating_example():
    # These are the ones with initial conditions:
    start_file = f"{PROJECT_PATH}/input-files/examples/lift_FINAL_NEW_strong.spectra"
    end_file = f"{PROJECT_PATH}/input-files/examples/lift_FINAL_NEW.spectra"

    trace_file = generate_filename(start_file, "_auto_violation.txt")
    delete_files(trace_file)
    generate_trace_asp(start_file, end_file, trace_file)
    self = Specification(start_file, trace_file, include_prev=False, random_hypothesis=True)
    self.run_pipeline()


def traffic_single_motivating_example():
    # These are the ones with initial conditions:
    start_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_FINAL_strong.spectra"
    end_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_FINAL.spectra"

    trace_file = generate_filename(start_file, "_auto_violation.txt")
    delete_files(trace_file)
    generate_trace_asp(start_file, end_file, trace_file)
    self = Specification(start_file, trace_file, include_prev=False, random_hypothesis=True)
    self.run_pipeline()


def traffic_motivating_example():
    # These are the ones with initial conditions:
    start_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_updated_FINAL_strong.spectra"
    end_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_updated_FINAL.spectra"

    trace_file = generate_filename(start_file, "_auto_violation.txt")
    delete_files(trace_file)
    generate_trace_asp(start_file, end_file, trace_file)
    self = Specification(start_file, trace_file, include_prev=False, random_hypothesis=True)
    self.run_pipeline()


def arbiter_motivating_example():
    # These are the ones with initial conditions:
    start_file = f"{PROJECT_PATH}/input-files/examples/Arbiter/Arbiter_FINAL_strong.spectra"
    end_file = f"{PROJECT_PATH}/input-files/examples/Arbiter/Arbiter_FINAL.spectra"

    trace_file = generate_filename(start_file, "_auto_violation.txt")
    delete_files(trace_file)
    generate_trace_asp(start_file, end_file, trace_file)
    self = Specification(start_file, trace_file, include_prev=False, random_hypothesis=True)
    self.run_pipeline()


def lift_trial():
    end_file = f"{PROJECT_PATH}/input-files/examples/lift_FINAL_normalised.spectra"
    start_file = f"{PROJECT_PATH}/input-files/examples/lift_FINAL_normalised_dropped.spectra"
    #
    # trace_file = generate_filename(start_file, "_auto_violation.txt")
    # delete_files(trace_file)
    # violation_file = generate_trace_asp(start_file, end_file, trace_file)
    # # outcome, learning_time = run_simulated_environment_jit(start_file, end_file)

    violation_file = generate_filename(start_file, "_auto_violation.txt")
    self = Specification(start_file, violation_file)
    self.run_pipeline()


def trial_traffic_48_43():
    start_file = f"{PROJECT_PATH}/input-files/examples/traffic_updated_FINAL_normalised_dropped48.spectra"
    end_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_updated_FINAL_normalised.spectra"
    outcome, learning_time = run_simulated_environment_jit(start_file, end_file, limit=10, include_prev=False)
    results = {}
    results[0] = (outcome, learning_time, 6, [])
    start_file = f"{PROJECT_PATH}/input-files/examples/traffic_updated_FINAL_normalised_dropped43.spectra"
    outcome, learning_time = run_simulated_environment_jit(start_file, end_file, limit=10, include_prev=False)
    results[1] = (outcome, learning_time, 9, [])
    df = results_to_df(results, end_file, types=["A,G"], include_prev=False)
    save_results_df(df)


def run_simulated_environment_jit(start_file, end_file, limit=-1, include_prev=False):
    trace_file = generate_filename(start_file, "_auto_violation.txt")
    delete_files(trace_file)
    count = 0
    learning_time = []
    while True:
        if count == limit:
            if self.is_realizable:
                return SimEnv.Realizable, learning_time
            return SimEnv.Timeout, learning_time
        violation_file, violated = generate_trace_asp(start_file, end_file, trace_file)
        if violation_file is None:
            return SimEnv.NoTraceFound, learning_time
        self = Specification(start_file, violation_file, violation_list=[], include_prev=include_prev)
        elapsed = self.run_pipeline()
        if elapsed is not None:
            learning_time.append(elapsed)
        count += 1
        # if self.realizable is None:
        #     return SimEnv.Timeout, learning_time
        if not self.violation_list:
            return SimEnv.Invalid, learning_time

        if not self.is_realizable:
            return SimEnv.Unrealizable, learning_time
        if semantically_identical(self.fixed_spec_file, end_file, assumptions_only=False):
            return SimEnv.Success, learning_time
        if semantically_identical(self.fixed_spec_file, end_file, assumptions_only=True):
            return SimEnv.IncorrectGuarantees, learning_time
        # start_file = self.fixed_spec_file


def rq1_random():
    files = {
        "Minepump":
            {
                "start": f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed0.spectra",
                "end": f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_fixed1.spectra"
            },
        "Arbiter":
            {
                "start": f"{PROJECT_PATH}/input-files/examples/Arbiter/Arbiter_FINAL_dropped0.spectra",
                "end": f"{PROJECT_PATH}/input-files/examples/Arbiter/Arbiter_FINAL.spectra"
            },
        "Lift":
            {  # "start": f"{PROJECT_PATH}/input-files/examples/lift_FINAL_NEW_strong.spectra",
                # "end": f"{PROJECT_PATH}/input-files/examples/lift_FINAL_NEW.spectra"
                "start": f"{PROJECT_PATH}/input-files/examples/lift_well_sep_strong.spectra",
                "end": f"{PROJECT_PATH}/input-files/examples/lift_well_sep.spectra"
            },
        "Traffic Single":
            {"start": f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_FINAL_strong.spectra",
             "end": f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_FINAL.spectra"
             },
        "Traffic":
            {"start": f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_updated_FINAL_strong.spectra",
             "end": f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_updated_FINAL.spectra"
             },
        "Genbuf":
            {"start": f"{PROJECT_PATH}/input-files/examples/genbuf_05_normalised_dropped.spectra",
             "end": f"{PROJECT_PATH}/input-files/examples/genbuf_05_normalised.spectra"
             }
    }
    timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")

    columns = ["name", "random_hypothesis", "trace_length", "n_assumptions", "n_guarantees", "n_violations",
               "av_variables_in_violations", "min_var_in_violations", "max_var_in_violations",
               "av_temporals_in_violations", "min_temp_in_violations", "max_temp_in_violations",
               "n_assumption_realizability_checks", "n_guarantee_realizability_checks", "av_min_solutions",
               "n_weakened_assumptions", "n_weakened_guarantees", "learning_time", "result", "file", "timestamp",
               "learner", "temporals"]

    n_runs = 0
    n_violation_traces = 10
    total_output = []
    learner = "ILASP"
    temporals = True
    runall = True
    strengthened = 5
    timeout = 100
    if FASTLAS:
        learner = "FastLAS"
    for name in files.keys():
        start_files = {files[name]["start"]: "non_temporal"}
        end_file = files[name]["end"]
        assert (realizable(end_file))
        if runall:
            filename = re.search(r"/([^/]*)\.spectra", files[name]["end"]).group(1)
            start_files = get_start_files(filename)  # , cap=strengthened * 4)
        start_count = 0
        for start_file in start_files.keys():
            if start_count == strengthened:
                continue
            if not realizable(start_file):
                continue

            trace_file = generate_filename(start_file, "_auto_violation.txt")
            delete_files(trace_file)
            violation = False
            for i in range(n_violation_traces):
                file, v = generate_trace_asp(start_file, end_file, trace_file)
                if v:
                    violation = True
            if not violation:
                continue
            start_count += 1
            violation_file = extract_nth_violation(trace_file, 0)
            assert (violation_file != "")
            output = []
            for i in range(n_runs):
                set_of_types = [True, False]
                if runall:
                    temp_type = start_files[start_file]
                    if temp_type == "temporal":
                        set_of_types = [True]
                    else:
                        set_of_types = [False]
                for include_prev in set_of_types:
                    if not temporals and include_prev and not runall:
                        continue
                    if name == "Genbuf" and include_prev and not FASTLAS:
                        continue
                    self = Specification(start_file, violation_file, random_hypothesis=True, include_prev=include_prev)
                    learning_time = self.run_pipeline(timeout_value=timeout)

                    assert (self.violation_list)
                    output_line = extract_stats_rq(end_file, learning_time, name, self, True)
                    output_line += [timestamp, learner, include_prev]
                    output.append(output_line)
                    df = pd.DataFrame([output_line], columns=columns)
                    df.to_csv("output-files/examples/latest_rq_results.csv", mode="a", header=False)

            for i in range(n_violation_traces):
                violation_file = extract_nth_violation(trace_file, i)
                if violation_file == "":
                    continue
                set_of_types = [True, False]
                if runall:
                    temp_type = start_files[start_file]
                    if temp_type == "temporal":
                        set_of_types = [True]
                    else:
                        set_of_types = [False]
                for include_prev in set_of_types:
                    if not temporals and include_prev and not runall:
                        continue
                    if name == "Genbuf" and include_prev and not FASTLAS:
                        continue
                    self = Specification(start_file, violation_file, random_hypothesis=False, include_prev=include_prev)
                    learning_time = self.run_pipeline(timeout_value=timeout)

                    assert (self.violation_list)
                    output_line = extract_stats_rq(end_file, learning_time, name, self, False)
                    output_line += [timestamp, learner, include_prev]
                    output.append(output_line)
                    df = pd.DataFrame([output_line], columns=columns)
                    df.to_csv("output-files/examples/latest_rq_results.csv", mode="a", header=False)
            total_output += output

    df = pd.DataFrame(total_output, columns=columns)
    outfile, n = get_last_results_csv_filename(1, "output-files/rq_results")
    df.to_csv(outfile)
    # df.to_csv("output-files/examples/latest_rq_results.csv")#,mode="a")


def iff_to_dnf(line):
    # line = '\tG((ENQ <-> DEQ) -> ((FULL <-> next(FULL)) &(EMPTY <-> next(EMPTY))));\n'
    if not re.search("<->", line):
        return line
    contents = list(parenthetic_contents_with_function(line))
    levels = max([tup[0] for tup in contents]) + 1
    for i in reversed(range(levels)):
        for j, par in enumerate(contents):
            if par[0] == i:
                # replacement bit
                one_up = [x for x in contents if x[0] == i + 1]
                string = par[2]
                for x in one_up:
                    string = re.sub(re.escape(x[2]), x[3], string)
                contents[j] = tuple(list(contents[j]) + [iff_to_dnf_sub(string)])
    for tup in contents:
        if tup[0] == 0:
            line = re.sub(re.escape(tup[2]), tup[3], line)
    return line
    # exp = re.search(r"(G|GF)\s*\((.*)<->(.*)\s*\)\s*;", line)
    # if (exp):
    #     a = exp.group(2)
    #     b = exp.group(3)
    #     return "\t" + exp.group(1) + "(" + a + "&" + b + "|" + negate(a) + "&" + negate(b) + ");\n\n"
    # return line


def iff_to_dnf_sub(formula):
    exp = re.search(r"(.*)<->(.*)", formula)
    if (exp):
        a = exp.group(1)
        b = exp.group(2)
        return "(" + a + "&" + b + "|" + negate(a) + "&" + negate(b) + ")"

    # formula = "((FULL & next(FULL)|!FULL&!next(FULL)) &(EMPTY & next(EMPTY)|!EMPTY&!next(EMPTY)))"

    return formula


def run_simulated_environment(start_file, end_file, limit=-1):
    trace_file = generate_filename(start_file, "_auto_violation.txt")
    delete_files(trace_file)
    count = 0
    learning_time = []
    while True:
        if count == limit:
            return SimEnv.Timeout, learning_time
        violation_file = generate_trace(start_file, end_file, trace_file)
        if violation_file is None:
            return SimEnv.Success, learning_time
        spec = Specification(start_file, violation_file, violation_list=[], include_prev=False)
        elapsed = spec.run_pipeline()
        if elapsed is not None:
            learning_time.append(elapsed)
        count += 1
        if not spec.is_realizable:
            return SimEnv.Unrealizable, learning_time
        if semantically_identical(spec.fixed_spec_file, end_file, assumptions_only=False):
            return SimEnv.Success, learning_time
        if semantically_identical(spec.fixed_spec_file, end_file, assumptions_only=True):
            return SimEnv.IncorrectGuarantees, learning_time
        # start_file = spec.fixed_spec_file


def example_case():
    spectra_file = f"{PROJECT_PATH}/input-files/examples/example_translation/example.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/examples/example_translation/example_violation.txt"
    violation_list = ["prev_and_next"]
    self = Specification(spectra_file, violation_file, violation_list, include_prev=False)
    self.format_spectra_file()
    self.generate_formula_df_from_spectra()
    self.run_pipeline()


def lift():
    spectra_file = f"{PROJECT_PATH}/input-files/examples/lift.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/examples/lift_violation_modified.txt"
    violation_list = ["button3_stays_on"]
    self = Specification(spectra_file, violation_file, violation_list, include_prev=False)
    # self.format_spectra_file()
    # self.spectra_to_DataFrames()

    self.run_pipeline()


def arbiter():
    spectra_file = f"{PROJECT_PATH}/input-files/examples/Arbiter/Arbiter_edited.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/examples/Arbiter/Arbiter_edited_violation_custom.txt"
    violation_list = ["a_follows_not_a"]
    self = Specification(spectra_file, violation_file, violation_list, include_prev=True)
    self.run_pipeline()


def genbuf_unsat_trial():
    spectra_file = "../example-files/genbuf_05_normalised_dropped0.spectra"
    violation_file = "../example-files/genbuf_05_normalised_dropped0_auto_violation_temp.txt"
    self = Specification(spectra_file, violation_file, include_prev=False)
    self.run_pipeline()


def traffic_guarantee_weaken_single():
    # Evaluation
    spectra_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_guarantee.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_guarantee_violation.txt"
    violation_list = ['car_moves_when_green']

    self = Specification(spectra_file, violation_file, violation_list, include_prev=False)
    self.run_pipeline()


def traffic_guarantee_weaken(include_prev, unguided_learning=False):
    spectra_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_guarantee_weaken.spectra"
    violation_file = f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_guarantee_weaken_violation.txt"
    violation_list = ['carA_moves_when_green']
    if unguided_learning:
        violation_list = []

    traffic = Specification(spectra_file, violation_file, violation_list, include_prev)
    traffic.run_pipeline()


def delete_all(folder):
    all_files = [os.path.join(folder, x) for x in os.listdir(folder)]
    folders = [x for x in all_files if os.path.isdir(x)]
    files = [x for x in all_files if x not in folders]
    delete_files(files)
    for folder in folders:
        delete_all(folder)


def run_all_case_studies(delete_broken=True, delete_old=True):
    # This removes previous runs
    start_time = time.asctime(time.localtime())
    if delete_old:
        delete_all(f"{PROJECT_PATH}/input-files/case-studies/modified-specs")
    path = f"{PROJECT_PATH}/input-files/case-studies/specifications/"
    folders = [path, path + "without_genuine/"]

    result_collection = {}
    for folder in folders:
        sub_folders = get_folders(folder, CASE_STUDY_EXCLUSION_LIST)
        for sub_folder in sub_folders:
            name = re.search(r"/([^/]*)$", sub_folder).group(1)
            print("\nRunning:\t" + name)
            result_collection[name] = run_case_study(sub_folder, start_time, delete_broken)
            print("End of " + name + "\n")
    # print_dict(result_collection)

    return result_collection


def run_case_study(folder, start_time, delete_broken=True):
    # folder = f"{PROJECT_PATH}/input-files/case-studies/specifications/simple_arbiter_v1/"
    files = translate_case_study(folder, start_time, delete_broken)
    filenames = [tup[1] for tup in files]
    outcome_dict = {}
    print("")
    for file in filenames:
        name = re.search("/([^/]*)\.spectra", file).group(1)
        print("Running Learning Procedure on " + name)
        outcome, string = run_procedure(file)
        outcome_dict[name] = (outcome, string)
    return outcome_dict


def find_case_study(case_study):
    # case_study = 'Arbiter_BC9'
    folder = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/"
    file = recursively_search(case_study, folder)
    return re.sub(re.escape("\\"), "/", file)


def run_translated_case_studies(realizable_specs=None):
    # with open("output-files/case-studies/case_study_translation.csv") as f:
    #     log = f.readlines()
    # log = pd.read_csv("output-files/case-studies/case_study_translation.csv")
    # log["seconds"] = log["Timestamp"].apply(to_seconds)
    # latest_log = log.loc[log["seconds"] == max(log["seconds"])]
    # latest_log.loc[log["Problem"] == "Realizable"]
    if realizable_specs is None:
        realizable_specs = {}

    # "Rg2_BC6"
    case_studies = list(realizable_specs.keys())
    if realizable_specs == {}:
        case_studies = ['Arbiter_BC9',
                        'Arbiter_BC19',
                        'Rg2_BC2',
                        'Rg2_BC6',
                        'Round-Robin_BC0',
                        'Round-Robin_BC3',
                        'Rrcs_BC1',
                        'Rrcs_BC12',
                        'Simple_Arbiter_Icse2018_Realizable_BC1',
                        'Simple_Arbiter_V1_BC23',
                        'Simple_Arbiter_V1_BC25',
                        'Simple_Arbiter_V2_BC5']

    outcome_dict = {}
    for case_study in case_studies:
        if realizable_specs == {}:
            filename = find_case_study(case_study)
            if filename == "file_not_found":
                continue
        else:
            filename = realizable_specs[case_study]
        print("Running Learning Procedure on " + case_study)
        outcome, string = run_procedure(filename)
        outcome_dict[case_study] = (outcome, string)
    return outcome_dict


def run_procedure(spectra_file, boundary_condition=True, include_prev=False):
    # spectra_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/RG2/Rg2_BC6.spectra"
    # spectra_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/without_genuine/rrcs/Rrcs_BC1.spectra"
    violation_file, violation_list = force_violation(spectra_file, boundary_condition)
    if violation_list == []:
        return Outcome.NO_VIOLATION_TRACE_FOUND, violation_file

    self = Specification(spectra_file, violation_file, violation_list, include_prev)
    # self.format_spectra_file()
    # self.spectra_to_DataFrames()
    # self.encode_ILASP()
    # self.run_ILASP()
    # run_clingo(spectra_file,False)

    # extract_df_content(self.formula_df, "guarantee1_1\n", "formula")

    self.run_pipeline()
    if self.is_realizable is None:
        run_clingo(spectra_file, False)
        return Outcome.UNEXPECTED_OUTCOME, ""
    if self.is_realizable:
        return Outcome.REALIZABLE_SPEC_GENERATED, ""
    if not self.is_realizable:
        return Outcome.NO_REALIZABLE_SPEC_REACHED, ""


def drop_and_run(limit=10, repeat=10, include_prev=False):
    files = [  # f"{PROJECT_PATH}/input-files/examples/Arbiter/Arbiter_FINAL.spectra",
        f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_FINAL.spectra",
        f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_FINAL.spectra",
        f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_updated_FINAL.spectra",
        # f"{PROJECT_PATH}/input-files/examples/lift_FINAL_NEW.spectra",
        f"{PROJECT_PATH}/input-files/examples/lift_FINAL.spectra"]
    # include_prevs = [False,True,True,False]
    assert (all([realizable(file, False) for file in files]))
    total_df = None
    for file in files:
        results1 = drop_and_evaluate(file, repeat, limit, expressions=["assumption"], include_prev=include_prev)
        if results1 is None:
            continue
        df = results_to_df(results1, file, types=["A"], include_prev=include_prev)
        if total_df is None:
            total_df = df
        else:
            total_df = pd.concat([total_df, df], ignore_index=True)
            save_results_df(total_df)
        results2 = drop_and_evaluate(file, repeat, limit, expressions=["assumption", "guarantee"],
                                     include_prev=include_prev)
        if results2 is None:
            total_df = pd.concat([total_df, df2], ignore_index=True)
            save_results_df(total_df)
            continue
        df2 = results_to_df(results2, file, types=["A,G"], include_prev=include_prev)
        total_df = pd.concat([total_df, df2], ignore_index=True)
        save_results_df(total_df)
    summarize_specs()


def translate_all_case_studies(delete_broken=True, delete_old=True, exclude=True, broad=False):
    # This removes previous runs
    start_time = time.asctime(time.localtime())
    if delete_old:
        delete_all(f"{PROJECT_PATH}/input-files/case-studies/modified-specs")
    path = f"{PROJECT_PATH}/input-files/case-studies/specifications/"
    folders = [path, path + "without_genuine/"]

    result_collection = {}
    for folder in folders:
        exclusions = CASE_STUDY_EXCLUSION_LIST
        if not exclude:
            exclusions = []
        sub_folders = get_folders(folder, exclusions)
        for sub_folder in sub_folders:
            name = re.search(r"/([^/]*)$", sub_folder).group(1)
            print("\nRunning:\t" + name)
            files = translate_case_study(sub_folder, start_time, delete_broken, broad)
            filenames = [tup[1] for tup in files]
            for file in filenames:
                spec_name = re.search(r"/([^/]*)\.spectra$", file).group(1)
                # spec_name = re.sub(r"_patterned$", "", spec_name)
                result_collection[spec_name] = file
            print("End of " + name + "\n")
    return result_collection


def save_copy_to(end_file, folder):
    name = re.search(r"/([^/]*)\.spectra", end_file).group(1)
    files = os.listdir(folder)
    prev_files = re.findall(name + r"(\d*)\.spectra", ' '.join(files))
    if len(prev_files) == 0:
        n = 0
    else:
        n = max([int(x) for x in prev_files]) + 1
    dest = folder + "/" + name + str(n) + ".spectra"
    shutil.copyfile(end_file, dest)
    return dest


def extract_stats_rq(end_file, learning_time, name, self, random_hypothesis):
    dest_file = save_copy_to(self.fixed_spec_file, "output-files/rq_files")
    assumptions = self.orig_formula_df.name.loc[self.orig_formula_df["type"] == "assumption"]
    guarantees = self.orig_formula_df.name.loc[self.orig_formula_df["type"] == "guarantee"]
    n_assumptions = len(assumptions)
    n_guarantees = len(guarantees)
    n_violations = len(self.violation_list)
    n_realizability_checks_ass = self.realizability_checks["assumption"]
    n_realizability_checks_gar = self.realizability_checks["guarantee"]
    av_min_solutions = mean(self.min_solutions)
    variables = strip_vars(self.orig_spec)
    n_var = []
    n_temp = []
    for viol_ass in self.violation_list:
        formula = extract_df_content(self.orig_formula_df, viol_ass, "formula")
        n_var.append(len(re.findall('|'.join(variables), formula)))
        n_temp.append(len(re.findall(r"G|F|next\(|PREV\(|prev\(|\||->|&", formula)))
    av_variables_violated = mean(n_var)
    av_temporals_violated = mean(n_temp)
    n_weakened_assumptions = n_weakened_expressions(assumptions, self)
    n_weakened_guarantees = n_weakened_expressions(guarantees, self)
    if name == "Genbuf":
        if self.is_realizable:
            result = "Realizable"
        else:
            result = "Unrealizable"
    else:
        result = semantically_identical_spot(self.fixed_spec_file, end_file)
        result = str(result).strip("SimEnv.")

    violation_file_length = max([int(x) for x in re.findall(",(\d*),", self.violation_trace)]) + 1
    output_line = [name, random_hypothesis, violation_file_length, n_assumptions, n_guarantees, n_violations,
                   av_variables_violated, min(n_var), max(n_var),
                   av_temporals_violated, min(n_temp), max(n_temp),
                   n_realizability_checks_ass, n_realizability_checks_gar, av_min_solutions, n_weakened_assumptions,
                   n_weakened_guarantees, learning_time, result, dest_file]
    return output_line


# TODO: BUG: formula_df == orig_formula_df
# Probably bcz formula_df updated only on initial run_pipeline
# or run_guarantee_weakening calls
def n_weakened_expressions(expressions, specification):
    return sum(map(lambda exp: extract_df_content(specification.formula_df, exp, "formula") != \
                               extract_df_content(specification.orig_formula_df, exp, "formula"),
                   expressions))


def mean(array):
    if len(array) == 0:
        average = 0
    else:
        average = sum(array) / len(array)
    return average


def run_unrealizable_spec(spectra_file):
    # spectra_file = f"{PROJECT_PATH}/input-files/Arbiter_raw.spectra"

    self = Specification(spectra_file, "", ["true_often"])
    self.format_spectra_file()
    self.generate_formula_df_from_spectra()
    # TODO: ask Titus why rename the fixed spec file when substituting pRespondsToS?
    # self.fixed_spec_file = pRespondsToS_substitution(self.spectra_file)
    self.encode_ILASP()

    self.check_realizability(learning_type=Learning.ASSUMPTION_WEAKENING, disable_log=True)


def print_case_study_simple(result_collection):
    for key in result_collection.keys():
        print(key)
        tup = result_collection[key]
        print(tup[0])
        print(tup[1])


def drop_random(end_file, expressions, n=1, include_prev=False, write=True):
    out_file = generate_filename(end_file, "_dropped.spectra")
    spec = read_file_lines(end_file)
    # pick a line
    poss_violates = [i + 1 for i, line in enumerate(spec) if
                     re.search("assumption|ass", line) and re.search(r"G", spec[i + 1]) and not re.search(r"F",
                                                                                                          spec[i + 1])]
    possible_ass = [i + 1 for i, line in enumerate(spec) if
                    re.search("assumption|ass", line) and re.search(r"G|GF", spec[i + 1])]
    possible_gar = [i + 1 for i, line in enumerate(spec) if
                    re.search("guarantee|gar", line) and re.search(r"G|GF", spec[i + 1])]

    possible_lines = poss_violates
    if len(expressions) > 1:
        possible_lines = possible_gar + possible_ass

    if len(poss_violates) == 0:
        return "none exist", n, ""
    if n > len(possible_lines):
        n = len(possible_lines)
    if n == 0:
        n = random.choice([x + 1 for x in range(len(possible_lines))])
    count = 0
    breach = 0
    # breach is failsafe if count keeps getting abused.
    while count < n and breach < 100:
        breach += 1
        if count == 0:
            i = random.choice(poss_violates)
        else:
            i = random.choice(possible_lines)
        count += 1
        line = spec[i]
        response = re.search(r'^\s*G\(([^-]*)->\s*F\((.*)\)\s*\)\s*;', line)
        liveness = re.search(r"GF\s*\((.*)\)\s*;", line)
        justice = re.search(r"G\s*\(([^F]*)\)\s*;", line)
        prefix = "\t("
        suffix = ");\n"
        if response:
            justice = False
            prefix = "\tG(" + response.group(1) + "F("
            line = response.group(2)
            if len(line.split("|")) < 2:
                count -= 1
                continue
            suffix = "));\n"
        if liveness:
            prefix = "\tGF("
            line = liveness.group(1)
        if justice:
            prefix = "\tG("
            line = justice.group(1)
        line = re.sub(";", "", line)
        disjuncts = line.split("|")
        if not include_prev:
            possibilities = [x for x in disjuncts if not re.search("PREV|next", x)]
            if len(possibilities) == 0:
                count -= 1
                continue
            r_choice = random.choice(possibilities)
            disjuncts = [x for x in disjuncts if x != r_choice]
        else:
            disjuncts.pop(random.choice([i for i in range(len(disjuncts))]))
        if len(disjuncts) == 0:
            spec[i] = ""
            spec[i - 1] = ""
        else:
            new_line = prefix + '|'.join(disjuncts) + suffix
            spec[i] = new_line

    if write:
        write_file(out_file, spec)
    else:
        out_file = ""
    return out_file, n, ''.join(spec)


# TODO: Modify this function to generate multiple violation traces, from which one can choose
# TODO: Consider DFS instead of BFS, to find a counter-strategy first, then look for similar others
def generate_trace(start_file: str, end_file: str, trace_file: str, max_depth: int = 5) -> Optional[str]:
    state_space, initial_state_space, legal_transitions = extract_transitions(end_file)
    state_space_s, initial_state_space_s, legal_transitions_s = extract_transitions(start_file)

    if state_space == state_space_s and initial_state_space == initial_state_space_s and legal_transitions == legal_transitions_s:
        return None

    # Prepare BFS of traces
    de: deque[(int, AnyNode, str)] = deque([])
    root: AnyNode = AnyNode(id="root")
    depth: int = 0

    trace: dict = {}
    timepoint: int = 0

    legal_starts, illegal_starts = [], []
    for x in initial_state_space:
        (legal_starts, illegal_starts)[x not in initial_state_space_s].append(x)
    for exp in illegal_starts:
        # Adding any final illegal state expression as
        AnyNode(id=exp, parent=root, exp=exp, end=True)
    for exp in legal_starts:
        # Add to queue for BFS
        de.append((depth, root, exp))

    last_state = random.choice(legal_starts)
    if len(illegal_starts) > 0:
        trace[timepoint] = illegal_starts[0]
    else:
        trace[timepoint] = last_state
    timepoint += 1

    # Generate one complete illegal trace
    while not illegal_starts:
        # if any transition exists that is legal in one and illegal in the other, do it and terminate, else do random transition
        possible_moves = legal_transitions[state_space.index(last_state)]
        # if last_state in state_space_s:
        possible_moves_s = legal_transitions_s[state_space_s.index(last_state)]
        illegal_moves = [x for x in possible_moves if x not in possible_moves_s]
        if len(illegal_moves) > 0:
            trace[timepoint] = state_space[random.choice(illegal_moves)]
            break
        last_state = state_space[random.choice(possible_moves)]
        trace[timepoint] = last_state
        timepoint += 1

    # Generate the tree of traces
    while GENERATE_MULTIPLE_TRACES and depth < max_depth:
        depth, cur_tree, last_state = de.popleft()
        new_tree = AnyNode(id=last_state, parent=cur_tree, exp=last_state, end=False)
        possible_moves = legal_transitions[state_space.index(last_state)]
        possible_moves_s = legal_transitions_s[state_space_s.index(last_state)]
        legal_moves, illegal_moves = [], []
        for x in possible_moves:
            (legal_moves, illegal_moves)[x not in possible_moves_s].append(x)
        for exp in illegal_starts:
            # Adding any final illegal state expression as
            AnyNode(id=exp, parent=new_tree, exp=exp, end=True)
        for exp in legal_starts:
            # Add to queue for BFS
            de.append((depth + 1, cur_tree, exp))

    write_trace(trace, trace_file)
    return trace_file


def force_violation(spectra_file, boundary_condition):
    # spectra_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/without_genuine/round-robin/Round-Robin_BC0.spectra"
    spec = read_file_lines(spectra_file)
    filter_string = "assumption"
    if boundary_condition:
        filter_string = "assumption -- negated_bc"
    timepoint = 0
    trace_name = "forced_violation"
    if boundary_condition:
        reg = re.search(r"(BC\d*)", spectra_file)
        if reg:
            trace_name += "_" + reg.group(1)
    # write initial conditions
    initial_formulas = [spec[i + 1] for i, line in enumerate(spec) if
                        line.find("--") >= 0 and gr_one_type(spec[i + 1])[1] == When.INITIALLY]

    output = assign_variables_to_trace(initial_formulas, spec, timepoint)
    if len(initial_formulas) > 0:
        timepoint += 1

    violation_list = []
    formulae = {}
    rules = {}
    broken_rules = {}
    for i, line in enumerate(spec):
        if re.search("assumption -- ", line):
            name = re.search(r"--\s*([^\s^:]*)", line).group(1)
            formula = re.sub(r"\s*", "", spec[i + 1])
            pResondsToS, when = gr_one_type(formula)
            if when == When.ALWAYS:
                formulae[name] = formula
                formula = re.sub(r"^G\(|\);", "", formula)
                parts = formula.split("->")
                if len(parts) == 1:
                    parts.insert(0, "")
                antecedent = re.sub(r"next", "X", parts[0])
                consequent = re.sub(r"next", "X", parts[1])
                rule = '|'.join([x for x in [negate_and_simplify(antecedent), consequent] if x != ""])
                # rule = negate_and_simplify(antecedent) + "|" + consequent
                rules[name] = rule
                if re.search(re.escape(filter_string), line):
                    consequent = negate_and_simplify(consequent)
                    rule = "&".join([x for x in [antecedent, consequent] if x != ""])
                    # rule = antecedent + "&" + consequent
                    broken_rules[name] = rule

    for name in broken_rules.keys():
        broken_rule = broken_rules[name]
        dnf_list = [value for key, value in rules.items() if key != name]
        conjuncts = extract_models_from_dnf(dnf_list)
        rule_components = [broken_rule + "&" + x for x in conjuncts]
        variables = strip_vars(spec)
        for rule in rule_components:
            contents = list(parenthetic_contents_with_function(rule))
            next_assignments = [tup[2] for tup in contents if tup[1] == "X"]
            current_assignments = rule
            for x in next_assignments:
                current_assignments = re.sub(re.escape(r"X(" + x + ")"), "", current_assignments)
            if conjunct_is_false('&'.join(next_assignments), variables):
                continue
            if conjunct_is_false(current_assignments, variables):
                continue
            # : Add prev functionality?
            #   may need to increment timepoints first if prev present
            # prev = [tup[2] for tup in contents if tup[1] == "V"]
            # output += assign_variables_to_trace(prev, spec, timepoint - 1)
            output += assign_variables_to_trace([current_assignments], spec, timepoint)
            output += assign_variables_to_trace(next_assignments, spec, timepoint + 1)
            output = re.sub(r"trace_name", trace_name, output)
            # line = "assumption -- assumption1:\n"
            violation_list.append(name)
            output_filename = generate_filename(spectra_file, "_violation.txt")
            write_file(output_filename, output)
            return output_filename, violation_list
    if len(broken_rules) > 0:
        error = "Cannot force violation of single assumption.\nThe following were attempted.\n\n"
    else:
        error = "No justice assumption found under '" + filter_string + "'"
    for name in broken_rules.keys():
        error += name + "\t:\t" + formulae[name] + "\n"
        error += "===========================================\n"
        error += "violation\t:\t" + broken_rules[name] + "\n"
        new_dict = {k: formulae[k] for k in formulae.keys() - {name}}
        error += dict_to_text(new_dict)
        error += "===========================================\n"
        error += "No model can satisfy these => no trace exists.\n"
    print(error)
    return error, []


def extract_models_from_dnf(dnf_list):
    # : enforce DNF
    disjuncts = [x.split("|") for x in dnf_list]
    disjuncts = [[x for x in disjunct if x != ""] for disjunct in disjuncts]
    curr_dis = [["X(" + x + ")" for x in disjunct] for disjunct in disjuncts if none_contain_X(disjunct)]
    [disjuncts.append(x) for x in curr_dis]
    # This operation could get large. perhaps do it manually one by one until a working rule is found
    conjuncts = list(product(*disjuncts))
    return ['&'.join(x) for x in conjuncts]


def assign_variables_to_trace(initial_formulas, spec, timepoint):
    variables = strip_vars(spec)
    assigned_vars = []
    output = ""
    for formula in initial_formulas:
        for var in variables:
            if re.search("!" + var, formula):
                assigned_vars.append(var)
                output += "not_holds_at(" + var + "," + str(timepoint) + ",trace_name).\n"
            elif re.search(var, formula):
                assigned_vars.append(var)
                output += "holds_at(" + var + "," + str(timepoint) + ",trace_name).\n"
        for var in assigned_vars:
            if var in variables:
                variables.remove(var)
    # If there are any variables left (not found in formulae) then set to false
    if len(initial_formulas) > 0:
        for var in variables:
            output += "not_holds_at(" + var + "," + str(timepoint) + ",trace_name).\n"
    return output


def remove_contradictions_strengthened():
    for folder in ["temporal", "non_temporal"]:
        path = f"{PROJECT_PATH}/input-files/strengthened/" + folder
        files = os.listdir(path)
        for file in [path + "/" + x for x in files]:
            if contains_contradictions(file, "assumption|asm"):
                delete_files(file)


def check_trivial_rq():
    df = pd.read_csv("../output-files/examples/latest_rq_results.csv")
    times = list(dict.fromkeys(df.timestamp))
    t = times[-1:]
    df = df.loc[[x in t for x in list(df.timestamp)]]
    triv_guar = {}
    by_name = {}

    for name in df.name.unique():
        by_name[name] = 0

    for file in df.file:
        name = list(df.loc[df.file == file].name)[0]
        triv_guar[file] = n_trivial_guarantees(file)
        by_name[name] += triv_guar[file]
        by_name[name] += triv_guar[file]
    print("Trivial Guar:")
    print_dict(by_name)

    output = pd.read_csv("../output-files/examples/latest_rq_results.csv")

    files = df.loc[df.n_guarantee_realizability_checks > 0].file
    for file in files:
        name = re.search(r"/([^/]*)_fixed", file).group(1)
        start_file = f"{PROJECT_PATH}/input-files/strengthened/temporal/" + name + ".spectra"
        end_gar = extract_raw_expressions("guarantee|gar", file)
        start_gar = extract_raw_expressions("guarantee|gar", start_file)
        n = sum([x not in end_gar for x in start_gar])
        output.loc[output.file == file, "n_weakened_guarantees"] = n

    output.to_csv("output-files/examples/latest_rq_results.csv")
    summarize_rq()

    sf = {}
    files = df.file
    for file in files:
        dest = re.sub("output-files/rq_files", "case_studies/fixed", file)
        shutil.copyfile(file, dest)
        name = re.search(r"/([^/]*)_fixed", file).group(1)
        start_file = f"{PROJECT_PATH}/input-files/strengthened/temporal/" + name + ".spectra"
        sf[name] = start_file

    for file in sf.values():
        dest = re.sub(f"{PROJECT_PATH}/input-files/strengthened/temporal", "case_studies/mutated", file)
        shutil.copyfile(file, dest)


def extract_raw_expressions(exp_type, file):
    spec = read_file_lines(file)
    variables = strip_vars(spec)
    spec = simplify_assignments(spec, variables)
    expressions = [re.sub(r"\s", "", spec[i + 1]) for i, line in enumerate(spec) if re.search("^" + exp_type, line)]
    expressions = [re.sub(r"pRespondsToS\((.*),(.*)\)", r"G(\1->F(\2))", x) for x in expressions]
    return expressions


def summarize_rq():
    df = pd.read_csv("../output-files/examples/latest_rq_results.csv")
    times = list(dict.fromkeys(df.timestamp))
    t = times[-1:]
    df = df.loc[[x in t for x in list(df.timestamp)]]

    learner_type = df.learner.unique()
    if len(learner_type) > 1:
        print("Multiple learners present")
        return
    learner = learner_type[0]

    replace = "REMOVE"

    values = [
        'trace_length',
        'n_assumptions', 'n_guarantees', 'n_violations',
        'av_variables_in_violations', 'min_var_in_violations',
        'max_var_in_violations', 'av_temporals_in_violations',
        'min_temp_in_violations', 'max_temp_in_violations',
        'n_assumption_realizability_checks', 'n_guarantee_realizability_checks',
        'av_min_solutions', 'n_weakened_assumptions', 'n_weakened_guarantees',
        'learning_time'
    ]

    col_map = {
        'name': 'Case-Study',
        'trace_length': "Trace Length",
        'n_assumptions': 'Asm',
        'n_guarantees': 'Gar',
        'n_violations': 'Violations',
        'n_assumption_realizability_checks': 'Asm',
        'n_guarantee_realizability_checks': 'Gar',
        'n_weakened_assumptions': 'Asm',
        'n_weakened_guarantees': 'Gar',
        'learning_time': 'Learning Time (s)',
        'av_min_solutions': replace,
        'count': 'Count',
        'min': 'Min',
        'max': 'Max',
        'mean': 'Mean',
        'av_temporals_in_violations': 'Operators',
        'min_temp_in_violations': 'Operators',
        'max_temp_in_violations': 'Operators',
        'av_variables_in_violations': 'Variables',
        'min_var_in_violations': 'Variables',
        'max_var_in_violations': 'Variables',
        'uccess': "Success"
    }

    viol_tab = [
        ('av_min_solutions', 'count'),
        ('trace_length', 'mean'),
        ('n_assumptions', 'mean'),
        ('n_guarantees', 'mean'),
        ('n_violations', 'mean'),

        ('av_temporals_in_violations', 'mean'),
        ('min_temp_in_violations', 'min'),
        ('max_temp_in_violations', 'max'),

        ('av_variables_in_violations', 'mean'),
        ('min_var_in_violations', 'min'),
        ('max_var_in_violations', 'max')
    ]

    perf_tab = [
        ('av_min_solutions', 'count'),
        ('n_assumption_realizability_checks', 'mean'),
        ('n_assumption_realizability_checks', 'min'),
        ('n_assumption_realizability_checks', 'max'),
        ('n_guarantee_realizability_checks', 'mean'),
        ('n_guarantee_realizability_checks', 'min'),
        ('n_guarantee_realizability_checks', 'max'),

        ('n_weakened_assumptions', 'mean'),
        ('n_weakened_guarantees', 'mean'),

        ('learning_time', 'mean'),
        ('learning_time', 'min'),
        ('learning_time', 'max')
    ]

    output = ""
    # for temporal in df.temporals.unique():
    #     for rand_hyp in df.random_hypothesis.unique():

    # for rand_hyp in [False]:
    #     for temporal in [True]:
    #
    rand_hyp = False
    temporal = True

    dftemp = df.loc[np.logical_and(df.temporals == temporal, df.random_hypothesis == rand_hyp)]

    ltime = dftemp.loc[:, ["name", "learning_time"]]
    ltime = ltime.rename(columns=col_map)

    ltime.boxplot(column="Learning Time (s)", by="Case-Study")
    plt.savefig("latex/rq/learning_time.pdf")

    df_pivot = dftemp.pivot_table(index="name", values=values,
                                  aggfunc=["mean", "min", "max", "count"], margins=True)
    df_pivot.columns = df_pivot.columns.swaplevel(0, 1)
    df_pivot.sort_index(axis=1, level=0, inplace=True)

    df_pivot = df_pivot.round(1)

    vio = df_pivot.loc[:, viol_tab]
    perf = df_pivot.loc[:, perf_tab]

    new_perf_cols = [('Realisability Checks', x[0], x[1]) if re.search("realizability", x[0]) else (replace, x[0], x[1])
                     for x in perf_tab]
    new_perf_cols = [('No. Weakened', x[1], x[2]) if re.search("n_weakened", x[1]) else x for x in new_perf_cols]
    perf.columns = pd.MultiIndex.from_tuples(new_perf_cols)

    new_vio_cols = [('In Violations', x[0], x[1]) if re.search("in_violations", x[0]) else (replace, x[0], x[1]) for x
                    in viol_tab]
    vio.columns = pd.MultiIndex.from_tuples(new_vio_cols)

    perf = perf.rename(columns=col_map)
    vio = vio.rename(columns=col_map)

    perf.index.names = ["Case-Study"]
    vio.index.names = ["Case-Study"]

    # if temporal:
    #     temp_type = " with temporal weakenings allowed, "
    # else:
    #     temp_type = " with no temporal weakenings, "

    if rand_hyp:
        viol_type = " fixed violation trace."
        # vio = vio.drop([('REMOVE', 'REMOVE', 'Count')], axis=1)
    else:
        viol_type = " different violation traces."

    cap = viol_type + " Learner: " + learner

    # perf = perf.T
    # vio = vio.T
    res = dftemp.pivot_table(index="name", columns="result", values="trace_length", aggfunc="count", margins=True)
    res = res.fillna(0)
    res = res.astype(int)
    res = res.rename(columns=col_map)
    res.columns.name = "Case-Study"
    res.index.names = ["Case-Study"]

    res_string = res.to_latex(caption="Results of Algorithm", position="bottom")
    perf_string = perf.to_latex(caption="Algorithm Performance", position="bottom",
                                multicolumn_format="c", multirow=True)
    vio_string = vio.to_latex(caption="Summary of mutated specifications and corresponding violating traces",
                              position="bottom", multicolumn_format="c", multirow=True)

    res_string = shrink_latex_table_width(res_string)
    perf_string = shrink_latex_table_width(perf_string)
    vio_string = shrink_latex_table_width(vio_string)

    res_string = reformat_latex_table_rq(res_string, key="Trace Length")
    perf_string = reformat_latex_table_rq(perf_string)
    vio_string = reformat_latex_table_rq(vio_string)

    # if temporal:
    #     output += perf_string + "\n\n"
    # else:
    output += vio_string + "\n\n" + perf_string + "\n\n" + res_string + "\n\n"

    write_file("../latex/rq/tables.txt", output)


def main():
    # arbiter_motivating_example()
    # minepump_motivating_example_thread_safe()
    minepump_motivating_example()
    # lift_motivating_example()
    # traffic_motivating_example()
    # lift()
    # arbiter()
    # traffic_guarantee_weaken_single()
    # minepump2_from_simulated_environment()
    # traffic_guarantee_weaken_single()
    # trial_unrealizable()
    # genbuf_simple()
    # strengthen_all()
    # rq1_random()
    # summarize_rq()
    # genbuf_unsat_trial()
    # remove_contradictions_strengthened()
    # drop_and_run()


if __name__ == '__main__':
    main()


def none_contain_X(strings):
    return all([x.find("X") < 0 for x in strings])


def format_name(spectra_file):
    return generate_filename(spectra_file, "_formatted.spectra")


def check_format(spectra_file):
    # TODO: ensure input file is well formatted
    return True
    # names required for assumptions/guarantees
    # no nesting of next/prev
    # no next/prev in liveness


def create_signature_from_file(spectra_file):
    variables = strip_vars(read_file_lines(spectra_file))
    output = "%---*** Signature  ***---\n\n"
    for var in variables:
        output += "atom(" + var + ").\n"
    output += "\n\n"
    return output


def has_extension(file_path, target_extension) -> bool:
    _, extension = os.path.splitext(file_path)
    return extension.lower() == target_extension.lower()


def run_clingo(clingo_file, return_violated_traces=False, exp_type="assumption"):
    assert has_extension(clingo_file, ".lp")
    if not return_violated_traces:
        print("Running Clingo to aid debugging")
    # This assumes my filepath and using WSL
    output = run_clingo_raw(clingo_file)
    output = output.split("\n")
    for i, line in enumerate(output):
        if len(line) > 100:
            output[i] = '\n'.join(line.split(" "))

    answer_set = generate_temp_filename(".answer_set")
    if return_violated_traces:
        return list(filter(re.compile(rf"violation_holds\(|{exp_type}\(|entailed\(").search, output))
    else:
        output = '\n'.join(output)
        write_file(answer_set, output)
        print(f"See file for output: {answer_set}")


def gr_one_type(formula):
    '''
    :param formula:
    :return: pRespondsToS, when
    '''
    formula = re.sub('\s*', '', formula)
    eventually = re.search(r"^GF", formula)
    pRespondsToS = PRS_REG.search(formula)
    initially = not re.search(r"^G", formula)
    if eventually:
        when = When.EVENTUALLY
    elif initially:
        when = When.INITIALLY
    elif pRespondsToS:
        when = When.EVENTUALLY
    else:
        when = When.ALWAYS
    return pRespondsToS, when


def drop_all(end_file):
    if not re.search("normalised", end_file):
        print("file not normalised!")
        exit(1)
    spec = read_file_lines(end_file)
    poss_violates = [i + 1 for i, line in enumerate(spec) if
                     re.search("assumption|ass", line) and re.search(r"G", spec[i + 1]) and not re.search(r"F",
                                                                                                          spec[i + 1])]
    possible_ass = [i + 1 for i, line in enumerate(spec) if
                    re.search("assumption|ass", line) and re.search(r"G|GF", spec[i + 1])]
    possible_gar = [i + 1 for i, line in enumerate(spec) if
                    re.search("guarantee|gar", line) and re.search(r"G|GF", spec[i + 1])]

    possible_lines = possible_ass + possible_gar
    possible_lines = [x for x in possible_lines if x not in poss_violates]
    response = re.compile(r'^\s*G\(([^-]*)->\s*F\((.*)\)\s*\)\s*;')
    liveness = re.compile(r"GF\s*\((.*)\)\s*;")
    justice = re.compile(r"G\s*\(([^F]*)\)\s*;")
    exp = re.compile(r"(G)\s*\(([^F]*)\)\s*;|(GF)\s*\((.*)\)\s*;")

    temporals = [exp.search(spec[n]).group(1) for n in possible_lines if exp.search(spec[n])]
    poss_drops = [exp.search(spec[n]).group(2).split("|") for n in possible_lines if exp.search(spec[n])]
    poss_dr_bool = [[bool(not re.search("next|PREV", i)) for i in x] for x in poss_drops]
    # total_drops = sum([sum(x) for x in poss_dr_bool])
    # all_variations = list(product(*[[True,False] for i in range(total_drops)]))
    all_variations = list(product(*[list(product(*[[y, False] if y else [y] for y in x])) for x in poss_dr_bool]))
    all_rules = [
        ['|'.join([poss_drops[i][j] for j, bool in enumerate(subtup) if not bool]) for i, subtup in enumerate(tup)] for
        tup in all_variations]
    [[temporals[i] + "(" + formula + ");\n" for i, formula in enumerate(x)] for x in all_rules]


def summarize_spec(spectra_file):
    # spectra_file = f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_FINAL.spectra"
    spec = read_file_lines(spectra_file)
    e_vars = strip_vars(spec, ["env"])
    s_vars = strip_vars(spec, ["sys"])
    assumptions = extract_all_expressions("assumption", spec)
    guarantees = extract_all_expressions("guarantee", spec)
    live = re.compile(r"^GF")
    safe = re.compile(r"^G[^F]*$")
    init = re.compile(r"^[^G]")
    resp = re.compile(r"^G[^F]*->F")
    regs = [init, safe, live, resp]

    columns = ["Type", "Variables", "Expressions", "Initial", "Safety", "Liveness", "Response", "Max Length (vars)",
               "Median Length (vars)"]
    env = summarize_exps(assumptions, e_vars, s_vars, regs)
    sys = summarize_exps(guarantees, s_vars, e_vars, regs)
    tot = summarize_exps(assumptions + guarantees, s_vars + e_vars, [], regs)
    df = pd.DataFrame([["$\mathcal(E)$"] + env, ["$\mathcal(S)$"] + sys, ["Total"] + tot], columns=columns)
    df.index = df["Type"]
    df = df.drop(columns=["Type"]).T
    return df


def summarize_exps(expressions, vars, other_vars, regs):
    output = [len(vars), len(expressions)]
    for reg in regs:
        output.append(count_reg(expressions, reg))

    if len(expressions) == 0:
        return output + [0, 0]

    exp_lengths = [len(re.findall(r"|".join(vars + other_vars), x)) for x in expressions]
    output.append(max(exp_lengths))
    output.append(median(exp_lengths))
    return output


def count_reg(string_list, reg_expr):
    return sum([bool(reg_expr.search(x)) for x in string_list])


def summarize_case_studies():
    csv_list = [110, 67, 66, 59, 52, 45, 38, 31, 30, 23, 19, 15, 14]
    outfolder = "output-files/results"
    df = None
    for n in csv_list:
        outfile = outfolder + "/output" + str(n) + ".csv"
        results = pd.read_csv(outfile, index_col=0)
        results["run_id"] = n
        if df is None:
            df = results
        else:
            df = pd.concat([df, results], axis=0)
    files = CASE_STUDY_FINALS
    keys = files.keys()
    df["Specification"] = [list(keys)[list(files.values()).index(file)] for file in df.file]
    df["Expressions"] = [re.sub(r"\[|\]|'", "", x) for x in df["types"]]
    df["Result"] = [re.sub(r"SimEnv\.", "", x) for x in df["outcome"]]
    df.to_csv("output-files/examples/output_summary.csv")


def summarize_specs():
    files = {  # f"{PROJECT_PATH}/input-files/examples/Arbiter/Arbiter_FINAL.spectra",
        "Lift": f"{PROJECT_PATH}/input-files/examples/lift_FINAL.spectra",
        "Lift New": f"{PROJECT_PATH}/input-files/examples/lift_FINAL_NEW.spectra",
        "Minepump": f"{PROJECT_PATH}/input-files/case-studies/modified-specs/minepump/genuine/minepump_FINAL.spectra",
        # TODO: Find this file
        "Traffic Single": f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_single_FINAL.spectra",
        "Traffic": f"{PROJECT_PATH}/input-files/examples/Traffic/traffic_updated_FINAL.spectra"}
    keys = files.keys()
    # header = pd.MultiIndex.from_product([list(keys), ["env", "sys", "Total"]], names=["Spec", "Type"])
    # list_of_df = [summarize_spec(file) for file in files.values()]
    # df = pd.concat(list_of_df, axis=1)
    # df.columns = header
    # write_file(df.to_latex(), "latex/summary.txt")

    result_file, n = get_last_results_csv_filename(shift=0)
    results = pd.read_csv(result_file, index_col=0)

    results = results[results["outcome"] != "SimEnv.Invalid"]

    results["Specification"] = [list(keys)[list(files.values()).index(file)] for file in results.file]
    results["Expressions"] = [re.sub(r"\[|\]|'", "", x) for x in results["types"]]
    results["Result"] = [re.sub(r"SimEnv\.", "", x) for x in results["outcome"]]

    lat = results_to_latex(results, rows=["Specification", "Result", "Expressions"])
    # lat = re.sub("rrr","|rrr",lat)
    lat += "\n\n" + results_to_latex(results, rows=["Result"])

    output = "\\usepackage{booktabs}\n\\begin{document}\n\\begin{table}\n\\centering\n" + lat + \
             "\\caption{my table}\n\\label{tab:my_label}\n\\end{table}\n\\end{document}\n"

    write_file("latex/output" + n + ".tex", output)


def results_to_latex(results, rows):
    final_df = results.pivot_table(index=rows, values=["total_time", "n_dropped"],
                                   aggfunc="mean", margins=True)
    count_df = results.pivot_table(index=rows, values=["outcome"],
                                   aggfunc="count", margins=True)
    final_df["total_time"] = final_df["total_time"].round(1)
    final_df["n_dropped"] = final_df["n_dropped"].round(1)
    lat = pd.concat([count_df, final_df], axis=1).to_latex()
    # lat = final_df.to_latex()
    # lat = re.sub(r"(\.\d)\d*\b", r"\1", lat)
    lat = re.sub(r"NaN", "-", lat)
    lat = re.sub(r"total\\_time", "Mean Learning", lat)
    lat = re.sub(r"n\\_dropped", "Mean Number of", lat)
    lat = re.sub(r"(" + rows[-1] + "\s*)&\s*&\s*&\s*", r"\1&&Strengthenings&Time (s)", lat)
    lat = re.sub("outcome", "Count", lat)
    lat = re.sub("& A", r"& $\\phi^\\mathcal{E}$", lat)
    lat = re.sub(",G", r",$\\phi^\\mathcal{S}$", lat)
    return lat


def formatted_mean(lst):
    if len(lst) == 0:
        return "-"
    av = sum(lst) / len(lst)
    return str(round(av, 2))


def extract_last_unrealizable_dropped():
    outfile, n = get_last_results_csv_filename(0)
    df = pd.read_csv(outfile)
    df["outcome"] == "SimEnv.Unrealizable"


def save_results_df(total_df):
    outfile, n = get_last_results_csv_filename()
    total_df.to_csv(outfile)


def get_last_results_csv_filename(shift=1, outfolder="output-files/results"):
    # outfolder = "output-files/results"
    all_files = os.listdir(outfolder)
    reg = re.compile(r"output(\d*).csv")
    if len(all_files) == 0:
        n = '0'
    else:
        last_file = max([int(reg.search(x).group(1)) for x in all_files if reg.search(x)])
        n = str(last_file + shift)
    outfile = outfolder + "/output" + n + ".csv"
    return outfile, n


def results_to_df(results, spectra_file, types, include_prev):
    columns = ["file", "run", "types", "prevs", "n_dropped", "outcome", "total_time", "max_time", "median_time",
               "n_runs"]
    output = []
    for key in results.keys():
        result = results[key]
        if len(result[1]) == 0:
            output.append([spectra_file, key, types, include_prev, result[2], result[0], 0, 0, 0, 0])
        else:
            output.append([spectra_file, key, types, include_prev, result[2], result[0], sum(result[1]), max(result[1]),
                           median(result[1]), len(result[1])])
    df = pd.DataFrame(output, columns=columns)
    return df


def increment_diff(diff, i, line):
    try:
        diff[i][line] += 1
    except KeyError:
        try:
            diff[i][line] = 1
        except KeyError:
            diff[i] = {}
            diff[i][line] = 1


def print_diff(final_spec, diff):
    keys = list(diff.keys())
    for i in keys:
        for line in diff[i].keys():
            final_spec[i] += "\t\t" + str(diff[i][line]) + ":" + line + "\n"
    print(''.join(final_spec))


def wrong_order(spec, final_spec):
    expr = re.compile("assumption -- |guarantee -- ")
    for i, line in enumerate(final_spec):
        if spec[i] != line and expr.search(line):
            return True
    return False


def re_order(spec, final_spec):
    final_ex = extract_expressions_to_dict(final_spec)
    start_ex = extract_expressions_to_dict(spec)
    spec = [x for x in spec if re.search("^sys|^env|^module|^spec", x)]
    for key in final_ex.keys():
        spec.append(str(key))
        spec.append(start_ex[key])
    return spec


def extract_expressions_to_dict(final_spec):
    expr = re.compile("assumption -- |guarantee -- ")
    expressions = {}
    for i, line in enumerate(final_spec):
        if expr.search(line):
            expressions[line] = final_spec[i + 1]
    return expressions


def satisfies(expression, state):
    disjuncts = expression.split("|")
    for disjunct in disjuncts:
        conjuncts = disjunct.split("&")
        if all([conjunct in state for conjunct in conjuncts]):
            return True
    return False


def transitions(state, state_space, primed_expressions, prevs):
    primed_expressions = [re.sub(r"PREV\((!*)([^\|]*)\)", r"\1prev_\2", x) for x in primed_expressions]
    # p_exp = primed_expressions[0]
    # p_exp.split("|")
    forced_expressions = [exp for exp in primed_expressions if
                          not any([variable in state for variable in exp.split("|")])]
    nexts = [re.search(r"next\((.*)\)", variable).group(1) for sublist in [exp.split("|") for exp in forced_expressions]
             for variable in sublist if re.search(r"next\((.*)\)", variable)]
    possible_next_states = [s for s in state_space if all([x in s for x in nexts])]
    new_states = possible_next_states
    for prev in prevs:
        var = re.search(r"prev_(.*)", prev).group(1)
        if "!" + var in state:
            new_states = [s for s in possible_next_states if prev not in s]
        else:
            new_states = [s for s in possible_next_states if prev in s]

    return [state_space.index(x) for x in new_states]


def transitions_jit(state, primed_expressions, unprimed_assignments, prevs):
    raise NotImplementedError("Transitions JIT is not finished and not implemented")

    primed_expressions = [re.sub(r"PREV\((!*)([^\|]*)\)", r"\1prev_\2", x) for x in primed_expressions]
    forced_expressions = [exp for exp in primed_expressions if
                          not any([variable in state for variable in exp.split("|")])]
    nexts = [re.search(r"next\((.*)\)", variable).group(1) for sublist in
             [exp.split("|") for exp in forced_expressions]
             for variable in sublist if re.search(r"next", variable)]
    possible_next_states = [s for s in state_space if all([x in s for x in nexts])]

    for prev in prevs:
        var = re.search(r"prev_(.*)", prev).group(1)
        if "!" + var in state:
            new_states = [s for s in possible_next_states if prev not in s]
        else:
            new_states = [s for s in possible_next_states if prev in s]

    return [state_space.index(x) for x in new_states]


def no_next(dis):
    conjuncts = dis.split("&")
    for conjunct in conjuncts:
        if re.search("next", conjunct):
            return False
    return True


def sub_next_only(dis):
    conjuncts = dis.split("&")
    output = '&'.join([re.sub(r"next\(([^\)]*)\)", r"\1", x) for x in conjuncts if re.search("next", x)])
    return re.sub(r"\(|\)", "", output)


def next_only(x, new_state):
    disjuncts = x.split("|")
    # TODO: seems to be generating things that violate assumptions.
    # disjuncts = [dis for dis in disjuncts if not no_next(dis)]
    # currs = [re.sub(r"next\([^\)]*\)", "", dis) for dis in disjuncts]
    # currs = [re.sub(r"&(\))|(\()&", r"\1",c) for c in currs]
    # currs = [re.sub(r"&&","&",c) for c in currs]
    #
    # [satisfies(c,new_state) for c in currs]

    disjuncts = [sub_next_only(dis) for dis in disjuncts if not no_next(dis)]
    return '|'.join(disjuncts)


def next_possible_assignments(new_state, primed_expressions_cleaned, primed_expressions_cleaned_s, unprimed_expressions,
                              unprimed_expressions_s, variables):
    unsat_next_exp = unsat_nexts(new_state, primed_expressions_cleaned)

    # TODO: VERYFY _s should mean the expressions have to be false in order for the violation to occur
    unsat_next_exp_s = unsat_nexts(new_state, primed_expressions_cleaned_s)

    if unsat_next_exp + unsat_next_exp_s + unprimed_expressions + unprimed_expressions_s == []:
        # Pick random assignment
        vars = [var for var in variables if not re.search("prev_", var)]
        i = random.choice(range(2 ** len(vars)))
        # TODO: replace i with 0 for deadlock - in order to make deterministic
        i = 0
        n = "{0:b}".format(i)
        assignments = '0' * (len(vars) - len(n)) + n
        assignments = [int(x) for x in assignments]
        state = ["!" + var if assignments[i] else var for i, var in enumerate(vars)]
        return [state], False
    return generate_model(unsat_next_exp + unprimed_expressions, unsat_next_exp_s + unprimed_expressions_s, variables,
                          force=True)

    # violation = unsat_next_exp + [negate(x) for x in unsat_next_exp_s]
    # parsed = parse_expr(re.sub(r"!", "~", '&'.join(violation)))
    # sympy.to_cnf(parsed)

    # next_assignments = possible_assignments(unsat_next_exp)
    # filtered_next = [x for x in next_assignments if
    #                  not any([satisfies(negate(expression), list(x)) for expression in unprimed_expressions])]
    #
    # violations = [x for x in next_assignments if
    #               any([satisfies(negate(expression), list(x)) for expression in unprimed_expressions_s])]
    # if len(violations) > 0:
    #     return violations, True
    # violations = [x for x in next_assignments if
    #               any([satisfies(negate(expression), list(x)) for expression in unsat_next_exp_s])]
    # if len(violations) > 0:
    #     return violations, True
    #
    # return filtered_next, False


def unsat_nexts(new_state, primed_expressions_cleaned):
    if new_state == []:
        return []
    unsat_primed_exp = [expression for expression in primed_expressions_cleaned if not satisfies(expression, new_state)]
    output = [next_only(x, new_state) for x in unsat_primed_exp]
    output = [x for x in output if x != ""]
    return output


def product_without_dupl(*args, repeat=1):
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] if y not in x else x for x in result for y in pool]  # here we added condition
    result = set(list(map(lambda x: tuple(sorted(x)), result)))  # to remove symmetric duplicates
    for prod in result:
        yield tuple(prod)


def split_conjuncts_and_remove_duplicates(tup):
    output = [item for sublist in [re.sub(r"^\(|\)$", r"", x).split("&") for x in list(tup)] for item in sublist]
    return tuple(dict.fromkeys(output))


def slow_state_space(var_space, unprimed_expressions):
    # TODO: check this?
    start = time.time()
    state_space = []
    assignment_sets = possible_assignments(unprimed_expressions)

    for i in range(2 ** len(var_space)):
        n = "{0:b}".format(i)
        assignments = '0' * (len(var_space) - len(n)) + n
        assignments = [int(x) for x in assignments]
        state = set([var[assignments[i]] for i, var in enumerate(var_space)])
        if any([s.issubset(state) for s in assignment_sets]):
            state_space.append(tuple(state))
        # if all([satisfies(expression, state) for expression in unprimed_expressions]):
        #     state_space.append(tuple(state))
    elapsed = (time.time() - start)
    print("Elapsed time: " + str(round(elapsed, 2)) + "s")
    return state_space


def contradiction(x):
    if len(x) == 0:
        return True
    x = list(dict.fromkeys(x))
    vars = [v.strip("!") for v in x]
    var = max(vars, key=vars.count)
    if len([y for y in vars if y == var]) > 1:
        return True
    return False


def possible_assignments(expressions):
    expressions = [x.split("|") for x in expressions]
    required_assignments = list(product_without_dupl(*expressions))
    required_assignments = [split_conjuncts_and_remove_duplicates(tup) for tup in required_assignments]
    assignment_sets = [set(x) for x in required_assignments if not contradiction(x)]
    return assignment_sets


def extract_transitions(file, assumptions_only=False):
    '''

    :param file:
    :return: state_space, initial_state_space, legal_transitions
    '''
    initial_expressions, prevs, primed_expressions, unprimed_expressions, variables = extract_expressions_from_file(file,
                                                                                                                    assumptions_only)

    var_space = [[x, "!" + x] for x in variables]
    if len(var_space) < 20:
        state_space = list(product(*var_space))
        state_space = [x for x in state_space if all([satisfies(expression, x) for expression in unprimed_expressions])]
    else:
        state_space = slow_state_space(var_space, unprimed_expressions)

    # assignment_set_initials = possible_assignments(initial_expressions)
    # initial_state_space = [x for x in state_space if
    #                        any([s.issubset(set(x)) for s in assignment_set_initials])]

    initial_state_space = [x for x in state_space if
                           all([satisfies(expression, x) for expression in initial_expressions])]
    legal_transitions = [transitions(state, state_space, primed_expressions, prevs) for state in state_space]
    return state_space, initial_state_space, legal_transitions


def semantically_identical(fixed_spec_file, end_file, assumptions_only):
    start_file = re.sub(r"_patterned", "", fixed_spec_file)
    state_space, initial_state_space, legal_transitions = extract_transitions(end_file, assumptions_only)
    state_space_s, initial_state_space_s, legal_transitions_s = extract_transitions(start_file, assumptions_only)

    if state_space == state_space_s and initial_state_space == initial_state_space_s and legal_transitions == legal_transitions_s:
        return True
    return False


def conjunct_is_false(string, variables):
    for var in variables:
        if re.search("!" + var, string) and re.search(r"[^!]" + var + r"|^" + var, string):
            return True
    return False


def last_state(trace, prevs, offset=0):
    prevs = ["prev_" + x if not re.search("prev_", x) else x for x in prevs]
    last_timepoint = max(re.findall(r",(\d*),", trace))
    if last_timepoint == "0" and offset != 0:
        return ()
    last_timepoint = str(int(last_timepoint) - offset)
    absent = re.findall(r"not_holds_at\((.*)," + last_timepoint, trace)
    atoms = re.findall(r"holds_at\((.*)," + last_timepoint, trace)
    assignments = ["!" + x if x in absent else x for x in atoms]
    if last_timepoint == '0':
        prev_assign = ["!" + x for x in prevs]
    else:
        prev_timepoint = str(int(last_timepoint) - 1)
        absent = re.findall(r"not_holds_at\((.*)," + prev_timepoint, trace)
        prev_assign = ["!" + x if x in absent else x for x in prevs]
    assignments += prev_assign
    variables = [re.sub(r"!", "", x) for x in assignments]
    assignments = [i for _, i in sorted(zip(variables, assignments))]
    return tuple(assignments)


def complete_deadlock_with_assignment(assignment, trace, name):
    end = f",{name})."
    last_timepoint = max(re.findall(r",(\d*),", trace))
    timepoint = str(int(last_timepoint) + 1)
    variables = [s for s in assignment if not re.search("prev_", s)]
    asp = ["not_holds_at(" + v[1:] if re.search("!", v) else "holds_at(" + v for v in variables]
    asp = [x + "," + timepoint + end for x in asp]
    return re.sub(r",[^,]*\)\.", end, trace) + '\n'.join(asp)


def to_seconds(t):
    return time.mktime(time.strptime(t))


def recursively_search(case_study, folder, exclusions=["genuine"]):
    folders = [os.path.join(folder, x) for x in os.listdir(folder) if x not in exclusions]
    files = [x for x in folders if not os.path.isdir(x)]
    sub_folders = [folder for folder in folders if folder not in files]
    for file in files:
        if file.find(case_study) >= 0:
            return file
    if len(sub_folders) == 0:
        return "file_not_found"
    for sub_folder in sub_folders:
        file_out = recursively_search(case_study, sub_folder)
        if file_out != "file_not_found":
            return file_out


def extract_nth_violation(trace_file, n):
    traces = read_file_lines(trace_file)
    trace = [x for x in traces if re.search("trace_name_" + str(n), x)]
    if len(trace) == 0:
        return ""
    temp_file = re.sub("auto_violation", "auto_violation_temp", trace_file)
    write_file(temp_file, trace)
    return temp_file


def contains_contradictions(start_file, exp_type):
    start_file = re.sub("_patterned\.spectra", ".spectra", start_file)
    start_exp = extract_all_expressions_spot(exp_type, start_file)
    linux_cmd = ["ltlfilt", "-f", f"{start_exp}", "--simplify"]
    output = subprocess.Popen(linux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
    output = output.decode('utf-8')
    reg = re.search(r"(\d)\n", output)
    if not reg:
        return False
    result = reg.group(1)
    return result == "0"


def get_start_files(name):  # , cap):
    output = {}
    for folder in ["temporal"]:
        long_folder = f"{PROJECT_PATH}/input-files/strengthened/" + folder
        files = os.listdir(long_folder)

        rel_files = re.findall(name + r"_dropped\d*\.spectra", '\n'.join(files))
        # if len(rel_files) > cap:
        #     rel_files = rel_files[:cap]
        for file in rel_files:
            output[long_folder + "/" + file] = folder
    return output


def n_trivial_guarantees(file):
    gar = extract_all_expressions_spot("guarantee|gar", file, True)
    count = 0
    for exp in gar:
        linux_cmd = "ltlfilt -f '" + exp + "' --simplify"
        output = \
            subprocess.Popen(["wsl"], stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                             stderr=subprocess.PIPE).communicate(
                input=linux_cmd.encode())[0]
        reg = re.search(r"b'(\d)\\n'", str(output))
        if reg and reg.group(1) == str(1):
            count += 1
    return count


def fix_df():
    df = pd.read_csv("../../output-files/examples/latest_rq_results.csv")
    np.isnan(df["temporals"])
    new_set = [np.isnan(x) for x in df["temporals"]]
    df_upper = df.loc[[not x for x in new_set]]
    df_lower = df.loc[new_set]

    df_lower = df_lower.drop(columns="Unnamed: 0.1")
    col_names = list(df_lower.columns)
    new_cols = col_names[1:] + ["REMOVE"]
    df_lower.columns = new_cols
    df_lower = df_lower.drop(columns="REMOVE")

    df_upper = df_upper.drop(columns=["Unnamed: 0.1", "Unnamed: 0"])

    output = pd.concat([df_lower, df_upper], axis=0)
    output.to_csv("output-files/examples/latest_rq_results.csv")


def reformat_latex_table_rq(perf_string, key="Count"):
    perf_string = re.sub(r"Case-Study[\s&]*\\*\n", "", perf_string)
    perf_string = re.sub(r"\{\}(\s*&\s*" + key + ")", r"Case-Study\1", perf_string)
    # perf_string = re.sub(r"\{\}(\s*&\s*Mean)", r"Case-Study\1", perf_string)
    perf_string = re.sub("REMOVE", " ", perf_string)
    return perf_string


def shrink_latex_table_width(perf_string):
    perf_string = re.sub(r"\n\\begin\{tabular\}", r"\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", perf_string)
    perf_string = re.sub(r"\n\\end\{tabular\}", r"\n\\end{tabular}\n}", perf_string)
    return perf_string


def generate_counter_strat(spec_file) -> Optional[List[str]]:
    cmd = ['java', '-jar', PATH_TO_CLI, '-i', spec_file, '--counter-strategy', '--jtlv']
    output = run_subprocess(cmd)
    if re.search("Result: Specification is unrealizable", output):
        output = str(output).split("\n")
        counter_strategy = list(filter(re.compile(r"\s*->\s*[^{]*{[^}]*").search, output))
        if PRINT_CS:
            print('\n'.join(counter_strategy))
        return counter_strategy
    elif re.search("FileNotFoundException", output):
        raise Exception(f"File {spec_file} doesn't exist!")
    elif re.search("Error:", output):
        raise Exception(output)

    return None
