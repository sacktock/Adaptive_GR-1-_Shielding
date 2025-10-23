import copy
import os.path
import re
import subprocess
from collections import OrderedDict, defaultdict
from typing import Set, Dict, Union, List, Optional

import pandas as pd

from spec_repair.enums import Learning, When, ExpType, SimEnv
from spec_repair.helpers.spectra_atom import SpectraAtom
from spec_repair.heuristics import choose_one_with_heuristic, manual_choice, HeuristicType
from spec_repair.ltl_types import CounterStrategy
from spec_repair.old.patterns import PRS_REG
from spec_repair.config import PROJECT_PATH, FASTLAS, PATH_TO_CLI, PATH_TO_TOOLBOX, PATH_TO_JVM
from spec_repair.old.specification_helper import strip_vars, assign_equalities, create_cmd, run_subprocess
from spec_repair.special_types import HoldsAtAtom
from spec_repair.util.file_util import read_file_lines, write_file, generate_temp_filename, write_to_file, \
    get_line_from_file

import threading
import jpype
import jpype.imports
import atexit
from jpype.types import *

if not jpype.isJVMStarted():
    jpype.startJVM(PATH_TO_JVM, "-ea", classpath=[f"{PATH_TO_TOOLBOX}:{PATH_TO_CLI}"])
    print("JVM started successfully")

SpectraToolbox = jpype.JClass('cores.SpectraToolbox')
SpectraCLI = jpype.JClass('tau.smlab.syntech.Spectra.cli.SpectraCliTool')

def pRespondsToS_substitution(output_filename):
    spec = read_file_lines(output_filename)
    found = False
    for i, line in enumerate(spec):
        line = line.strip("\t|\n|;")
        if PRS_REG.search(line):
            found = True
            s = re.search(r"G\(([^-]*)", line).group(1)
            p = re.search(r"F\((.*)", line).group(1)
            if p[-2:] == "))":
                p = p[0:-2]
            else:
                print("Trouble extracting p from: " + line)
                exit(1)
                # return "No_file_written:" + line
            replacement = "\tpRespondsToS(" + s + "," + p + ");\n"
            spec[i] = replacement
    if found:
        spec.append(''.join(read_file_lines(f"{PROJECT_PATH}/files/pRespondsToS.txt")))
        new_filename = generate_temp_filename('.spectra')
        write_file(new_filename, spec)
        return new_filename
    return output_filename


# TODO: toggle "sorted" off when performance optimised
def create_signature(spec_df: pd.DataFrame):
    variables = extract_variables(spec_df)
    output = "%---*** Signature  ***---\n\n"
    for var in sorted(variables):
        output += f"atom({var}).\n"
    output += "\n\n"
    return output


def create_atom_signature_asp(spec_atoms: Set[SpectraAtom]):
    output = "%---*** Signature  ***---\n\n"
    for atom in sorted(spec_atoms):
        output += f"atom({atom.name}).\n"
    output += "\n\n"
    return output


def extract_variables(spec_df: pd.DataFrame) -> List[str]:
    variables = set()
    for _, row in spec_df.iterrows():
        antecedents: List[Dict[str, List[str]]] = parse_formula_str(row['antecedent'])
        consequents: List[Dict[str, List[str]]] = parse_formula_str(row['consequent'])

        for conjunction in antecedents + consequents:
            for assignments in conjunction.values():
                for assignment in assignments:
                    variables.add(assignment.split("=")[0].strip())

    return list(variables)


class CSTraces:
    trace: str
    raw_trace: str
    is_deadlock: bool

    def __init__(self, trace, raw_trace, is_deadlock):
        self.trace = trace
        self.raw_trace = raw_trace
        self.is_deadlock = is_deadlock


def cs_to_cs_trace(cs: CounterStrategy, cs_name: str, heuristic: HeuristicType) -> CSTraces:
    trace_name_dict: dict[str, str] = cs_to_named_cs_traces(cs)
    cs_trace_raw, cs_trace_path = choose_one_with_heuristic(list(trace_name_dict.items()), heuristic)
    cs_trace = trace_replace_name(cs_trace_raw, cs_trace_path, cs_name)
    is_deadlock = "DEAD" in cs_trace_path
    return CSTraces(cs_trace, cs_trace_raw, is_deadlock)


def cs_to_named_cs_traces(cs: CounterStrategy) -> dict[str, str]:
    start = "INI"
    output = ""
    trace_name_dict: dict[str, str] = {}
    extract_trace(cs, output, start, 0, "ini", trace_name_dict)

    return trace_name_dict


def trace_replace_name(trace: str, old_name: str, new_name: str) -> str:
    reg = re.compile(rf"\b{old_name}\b")
    trace = reg.sub(new_name, trace)
    trace = re.sub(rf"(trace\({new_name})", rf"% CS_Path: {old_name}\n\n\1", trace)
    return trace


# TODO: generate multiple counter-strategies
def create_cs_traces(ilasp, learning_type: Learning, cs_list: List[CounterStrategy]) \
        -> Dict[str, CSTraces]:
    count = 0
    traces_dict: dict[str, CSTraces] = {}
    for lines in cs_list:
        trace_name_dict = cs_to_named_cs_traces(lines)
        cs_trace, cs_trace_path = choose_one_with_heuristic(list(trace_name_dict.items()), manual_choice)
        cs_trace_list = [cs_trace]
        # TODO: make it clear that a single trace/name pair is created for each element in the list
        trace, trace_names = create_trace(cs_trace_list, ilasp=ilasp, counter_strat=True,
                                          learning_type=learning_type)
        replacement = rf"counter_strat_{count}"
        for name in trace_names:
            trace = trace_replace_name(trace, name, replacement)
        count += 1
        # Add trace to counter-strat collection:
        is_deadlock = "DEAD" in cs_trace_path
        traces_dict[replacement] = CSTraces(trace, cs_trace, is_deadlock)

    return traces_dict


def create_trace(violation_file: Union[str, List[str]], ilasp=False, counter_strat=False,
                 learning_type=Learning.ASSUMPTION_WEAKENING):
    # This is for starting with unrealizable spec - an experiment
    if violation_file == "":
        return ""
    if type(violation_file) is not list:
        trace = read_file_lines(violation_file)
    else:
        trace = violation_file
    trace = re.sub("\n+", "\n", '\n'.join(trace)).split("\n")
    output = "%---*** Violation Trace ***---\n\n"
    trace_names: Set[str] = set(map(lambda match: match.group(1),
                                    filter(None,
                                           map(lambda line: re.search(r",\s*([^,]*)\)\.", line),
                                               trace)
                                           )
                                    )
                                )
    for name in trace_names:
        reg = re.compile(re.escape(name))
        sub_trace = list(filter(reg.search, trace))

        # TODO: understand infinite traces & use to rework counter-strategy trees
        # TODO: replace is_infinite with Sx->Sy->Sx->Sy and not "DEAD" in name
        is_infinite = bool(re.search("ini_S\d", name))
        # This is for making counter strategies positive when guarantee weakening:
        if learning_type == Learning.GUARANTEE_WEAKENING:
            pos_int = False
        else:
            pos_int = counter_strat
        output = create_pos_interpretation(ilasp, output, sub_trace, is_infinite, pos_int)
    if counter_strat:
        return output, trace_names
    else:
        return output


def create_pos_interpretation(ilasp: bool, output: str, trace: List[str], is_infinite: bool,
                              counter_strat: bool) -> str:
    max_timepoint = 0
    for line in trace:
        line = re.sub(r"\s", "", line)
        timepoint = line.split(",")[-2]
        max_timepoint = max(max_timepoint, int(timepoint))
    # TODO: understand why violation name is the last line of the trace
    violation_name = trace[-1].split(",")[-1].replace(").", "")
    if is_infinite:
        states = violation_name.split("_")
        state_count = [states.count(i) for i in states]
        if 2 in state_count:
            loop = state_count.index(2)
        else:
            is_infinite = False
    if ilasp and not counter_strat:
        output += "#pos({entailed(" + violation_name + ")},{},{\n"
    if ilasp and counter_strat:
        output += "#pos({},{entailed(" + violation_name + ")},{\n"
    output += f"trace({violation_name}).\n\n"
    output += create_time_fact(max_timepoint + 1, "timepoint", [0, violation_name])
    output += create_time_fact(max_timepoint, "next", [1, 0, violation_name])
    if is_infinite:
        output += create_time_fact(1, "next", [loop, max_timepoint, violation_name])
    output += '\n' + '\n'.join(trace) + '\n'
    if ilasp:
        output += "\n}).\n\n"
    return output


def trace_list_to_ilasp_form(asp_trace: str, learning: Learning) -> str:
    output = "%---*** Violation Trace ***---\n\n"
    asp_trace = asp_trace.split('\n')
    individual_traces = get_individual_traces(asp_trace)
    for trace in individual_traces:
        output += trace_single_asp_to_ilasp_form(trace, learning)
    return output


def trace_list_to_asp_form(traces: List[str]) -> str:
    output = "%---*** Violation Trace ***---\n\n"
    traces = remove_multiple_newlines(traces)
    individual_traces = get_individual_traces(traces)
    for trace in individual_traces:
        output += trace_single_to_asp_form(trace)
    return output


def get_individual_traces(traces: List[str]) -> List[List[str]]:
    """
    There may be multiple states of traces of different names.
    We isolate them based on their names
    """
    individual_traces = []
    trace_names: Set[str] = get_trace_names(traces)
    for name in trace_names:
        sub_trace = isolate_trace_of_name(traces, name)
        individual_traces.append(sub_trace)
    return individual_traces


def isolate_trace_of_name(trace: List[str], name: str):
    """
    There may be multiple states of traces of different names.
    We have the names, now we only need to isolate the specific
    individual trace by its name.
    e.g. names: trace_name_0, ini_S0_S1, ini_S0_S1_S1
    """
    reg = re.compile(re.escape(name))
    sub_trace = list(filter(reg.search, trace))
    return sub_trace


def get_trace_names(trace: List[str]) -> Set[str]:
    return set(map(lambda match: match.group(1),
                   filter(None,
                          map(lambda line: re.search(r",\s*([^,]*)\)\.", line),
                              trace)
                          )
                   )
               )


def trace_single_to_asp_form(trace: List[str]) -> str:
    max_timepoint = 0
    for line in trace:
        line = re.sub(r"\s", "", line)
        timepoint = line.split(",")[-2]
        max_timepoint = max(max_timepoint, int(timepoint))
    # TODO: understand why violation name is the last line of the trace
    violation_name = trace[-1].split(",")[-1].replace(").", "")
    output = f"trace({violation_name}).\n\n"
    output += create_time_fact(max_timepoint + 1, "timepoint", [0, violation_name])
    output += create_time_fact(max_timepoint, "next", [1, 0, violation_name])

    output += complete_loop_if_necessary(violation_name, max_timepoint)
    output += '\n' + '\n'.join(trace) + '\n'
    return output


def trace_single_asp_to_ilasp_form(trace: List[str], learning: Learning) -> str:
    """
    Pre: a single trace, with a single name, is provided
    """
    name = get_trace_names(trace).pop()
    raw_pattern = r'ini_(S\d+)_.*'
    cs_pattern = r'counter_strat_\d+'
    is_counter_strat: bool = bool(re.match(raw_pattern, name) or re.match(cs_pattern, name))
    if learning == Learning.ASSUMPTION_WEAKENING and is_counter_strat:
        output = f"#pos({{}},{{entailed({name})}},{{\n"
    else:
        output = f"#pos({{entailed({name})}},{{}},{{\n"
    output += '\n' + '\n'.join(trace) + '\n}).\n'
    return output


def complete_loop_if_necessary(violation_name, max_timepoint) -> str:
    states = get_state_numbers(violation_name)
    if not states:
        return ""
    max_state = max(states)
    state_timepoint_diff = max_timepoint - max_state
    match states[-2:]:
        case [s1, s2]:
            if s1 >= s2:
                return f"next({s2 + state_timepoint_diff},{s1 + state_timepoint_diff},{violation_name}).\n"
    return ""


def get_state_numbers(name: str) -> List[int]:
    """
    Extract numeric values of ini_S1_S2_...SN
    """
    pattern = r'ini_(S\d+)_.*'
    match = re.match(pattern, name)

    if match:
        numbers_list = re.findall(r'\d+', name)
        return [int(num) for num in numbers_list]
    return []


def create_time_fact(max_timepoint, name, param_list=None):
    if param_list is None:
        param_list = []
    output = ""
    for i in range(max_timepoint):
        strings = [str(i + x) if type(x) == int else x for x in param_list]
        output += f"{name}({','.join(strings)}).\n"
    return output


# TODO: replace traces as Dict with a Set[Tuple[str,str]]
def extract_trace(lines, output, start, timepoint, trace_name, traces: Dict[str, str]) -> Optional[str]:
    if len(re.findall(start, trace_name)) > 1 or start == "DEAD":
        output = re.sub("trace_name", trace_name, output)
        return output
    pattern = re.compile("^" + start)
    states = list(filter(pattern.search, lines))
    env = re.compile("{(.*)}\s*/", ).search(states[0]).group(1)
    output += vars_to_asp(env, timepoint)
    for state in states:
        sys = re.compile("/\s*{(.*)}", ).search(state).group(1)
        out_copy = copy.deepcopy(output)
        out_copy += vars_to_asp(sys, timepoint)
        next = extract_string_within("->\s*([^\s]*)\s", state)
        new_trace_name = trace_name + "_" + next
        new_output = extract_trace(lines, out_copy, next, timepoint + 1, new_trace_name, traces)
        if new_output is not None:
            traces[new_output] = new_trace_name


def vars_to_asp(sys, timepoint) -> str:
    vars = re.split(",\s*", sys)
    output = "\n".join([var_to_asp(var, timepoint) for var in vars])
    output += "\n"  # TODO: consider removing this last line
    return output


def var_to_asp(var, timepoint) -> str:
    parts = var.split(":")
    suffix = ""
    if parts[1] == "false":
        suffix = "not_"
    params = [parts[0], str(timepoint), "trace_name"]
    return f"{suffix}holds_at({','.join(params)})."


def extract_string_within(pattern, line, strip_whitespace=False):
    line = re.compile(pattern).search(line).group(1)
    if strip_whitespace:
        return re.sub(r"\s", "", line)
    return line


def get_assumptions_and_guarantees_from(start_file) -> pd.DataFrame:
    spec: List[str] = format_spec(read_file_lines(start_file))
    spec_df: pd.DataFrame = spectra_to_df(spec)
    return spec_df


def format_spec(spec):
    variables = strip_vars(spec)
    spec = word_sub(spec, "spec", "module")
    spec = word_sub(spec, "alwEv", "GF ( ")
    spec = word_sub(spec, "alw", "G ( ")
    # 'I' is later removed as not real Spectra syntax:
    spec = word_sub(spec, "ini", "I ( ")
    spec = word_sub(spec, "asm", "assumption --")
    spec = word_sub(spec, "gar", "guarantee --")
    # This bit deals with multivalued 'enums'
    spec, new_vars = enumerate_spec(spec)
    for i, line in enumerate(spec):
        words = line.strip("\t").split(" ")
        words = [x for x in words if x != ""]
        # This bit fixes boolean style
        if words and words[0] not in ['env', 'sys', 'spec', 'assumption', 'guarantee', 'module']:
            if len(re.findall(r"\(", line)) == len(re.findall(r"\)", line)) + 1:
                line = line.replace(";", " ) ;")
            # This replaces next(A & B) with next(A) & next(B):
            line = spread_temporal_operator(line, "next")
            line = spread_temporal_operator(line, "PREV")
            line = assign_equalities(line, variables + new_vars)
            spec[i] = line
    # This simplifies multiple brackets to single brackets
    # spec = [re.sub(r"\(\s*\((.*)\)\s*\)", r"(\1)", x) for x in spec]
    spec = [remove_trivial_outer_brackets(x) for x in spec]
    # This changes names that start with capital letters to lowercase so that ilasp/clingo knows they are not variables.
    spec = [re.sub('--[A-Z]', lambda m: m.group(0).lower(), x) for x in spec]
    return spec


def enumerate_spec(spec):
    new_vars = []
    for i, line in enumerate(spec):
        line = re.sub(r"\s", "", line)
        words = line.split(" ")
        reg = re.search(r"(env|sys){", line)
        if reg:
            # if words[0] in ['env', 'sys'] and line.find("{") >= 0:
            enum = extract_string_within("{(.*)}", line, True).split(",")
            name = extract_string_within("}(.*);", line, True)
            for value in enum:
                pattern = f"{name}\s*=\s*{value}"
                replacement = f"{name}_{value}"
                new_vars.append(replacement)
                spec = [re.sub(pattern, replacement, x) for x in spec]
                pattern = pattern.replace("=", "!=")
                replacement = f"!{replacement}"
                spec = [re.sub(pattern, replacement, x) for x in spec]
            replacement_line = ""
            for var in new_vars:
                replacement_line += reg.group(1) + " boolean " + var + ";\n\n"
            spec[i] = replacement_line
    return spec, new_vars


def spread_temporal_operator(line, temporal):
    pattern = r"(!)?" + temporal + r"\(([^\)]*)(&|\|)\s*"
    replacement = temporal + r"(\1\2) \3 \1" + temporal + "("
    while re.search(pattern, line):
        line = re.sub(pattern, replacement, line)
    line = re.sub("!" + temporal + r"\(", temporal + "(!", line)
    return line


def word_sub(spec: list[str], word: str, replacement: str):
    """
    Takes every expression in a spec and substitute every word in it with another
    :param spec: Specification as list of strings
    :param word: (recurrent) word to be replaced
    :param replacement: Word to replace by
    :return: new_spec.
    """
    return [re.sub(f"\b{word}\b", replacement, x) for x in spec]


def remove_trivial_outer_brackets(output):
    if has_trivial_outer_brackets(output):
        return output[1:-1]
    return output


def has_trivial_outer_brackets(output):
    contents = list(parenthetic_contents(output))
    if len(contents) == 1:
        if len(contents[0][1]) == len(output) - 2:
            return True
    return False


def parenthetic_contents(text):
    """Generate parenthesized contents in string as pairs (level, contents)."""
    stack = []
    for i, c in enumerate(text):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            yield (len(stack), text[start + 1: i])


# TODO: Revisit when optimising
def illegal_assignments(spec_df: pd.DataFrame, violations, trace):
    illegals = dict()
    # Violations needs to contain some values
    if trace == "" or not violations:
        return illegals
    expression_names: List[str] = re.findall(r"[assumption|guarantee]\(([^\)^,]*)\)", violations[0])
    for exp_name in expression_names:
        when = extract_df_content(spec_df, exp_name, "when")
        violated_timepoints = re.findall(r"violation_holds\(" + exp_name + r",(\d*),([^\)]*)\)", violations[0])
        preds: List[str] = []
        for vt in violated_timepoints:
            preds += extract_predicates(vt, trace)
        if when == when.EVENTUALLY:
            preds: List[str] = [re.sub(r"at_next\(|at_prev\(|at\(", "at_eventually(", x) for x in preds]
        preds = list(dict.fromkeys(preds))
        preds = [re.sub(r"\.", "", x) for x in preds]
        negs = [x[4:] if re.search(r"^not_", x) else "not_" + x for x in preds]
        illegals[exp_name] = negs
        # illegals[exp_name] = [x for x in negs if x not in preds]
    return illegals


def extract_df_content(formula_df: pd.DataFrame, name: str, extract_col: str):
    try:
        extracted_item = formula_df.loc[formula_df["name"] == name, extract_col].iloc[0]
        return extracted_item
    except IndexError:
        print(f"Cannot find name:\t'{name}'\n\nIn specification expression names:\n")
        print(formula_df["name"])
        exit(1)


def extract_predicates(vt, trace):
    trace_list = trace.split("\n")
    unprimed_preds = extract_preds_at(trace_list, vt, 0)
    prev_preds = extract_preds_at(trace_list, vt, -1)
    next_preds = extract_preds_at(trace_list, vt, 1)
    return unprimed_preds + prev_preds + next_preds


def extract_preds_at(trace_list, vt, offset):
    timepoint_string = "," + str(int(vt[0]) + offset) + "," + vt[1]
    swap = ""
    if offset == -1:
        swap = "_prev"
    if offset == 1:
        swap = "_next"
    preds = [re.sub(r"at\(", "at" + swap + "(", x) for x in trace_list if re.search(r"_.*" + timepoint_string, x)]
    return [re.sub(timepoint_string, ",V1,V2", x) for x in preds]


def remove_multiple_newlines(text):
    return re.sub("\n+", "\n", '\n'.join(text)).split("\n")


def replace_false_true(string):
    return string.replace("false", "__PLACEHOLDER__").replace("true", "false").replace("__PLACEHOLDER__", "true")


def flip_assignments(assignments: list[str]) -> list[str]:
    return [replace_false_true(assignment) for assignment in assignments]


def integrate_rule(temp_op, conjunct, learning_type, line: str):
    expression = extract_content_of_invariant(line)
    antecedent, consequent = extract_antecedent_and_consequent(expression)
    conjunct = re.sub("\s", "", conjunct)
    facts = conjunct.split(";")
    if FASTLAS:
        facts = [x for x in facts if x != ""]
    is_eventually_consequent = bool(re.match(r"^F\(.+\)", consequent))
    flip = learning_type == Learning.ASSUMPTION_WEAKENING or is_eventually_consequent
    assignments = extract_assignments_from_facts(facts, flip)

    if learning_type == Learning.ASSUMPTION_WEAKENING or is_eventually_consequent:
        return integrate_antecedent(temp_op, assignments, antecedent, consequent)

    if learning_type == Learning.GUARANTEE_WEAKENING:
        return integrate_consequent(temp_op, assignments, antecedent, consequent)


def extract_content_of_invariant(line: str) -> str:
    match = re.search(r'G\((.*)\);?', line)
    assert match
    content_inside = match.group(1).strip()
    return content_inside


def extract_antecedent_and_consequent(line: str) -> tuple[Optional[str], str]:
    match = re.match(r'^(.*?)\s*->\s*(.*)$', line)
    if match:
        antecedent = match.group(1)
        consequent = match.group(2)
        return antecedent, consequent
    else:
        # If there's no "->", we assume content is just the consequent
        return None, line


def integrate_antecedent(temp_op, assignments, antecedent, consequent):
    # next_assignments = [x for i, x in enumerate(assignments) if re.search("next", facts[i])]
    # cur_assignments = [x for x in assignments if x not in next_assignments]
    cur_assignments = assignments
    if antecedent:
        op = "G"
        head = antecedent
    else:
        op = "G"
        head = ""
    disjuncts = get_disjuncts(head)
    amended_disjuncts = []
    for disjunct in disjuncts:
        conjuncts = get_conjuncts(disjunct)
        # next_conjuncts = [x for x in conjuncts if re.search("next", x)] + next_assignments
        # cur_conjuncts = [x for x in conjuncts if x not in next_conjuncts] + cur_assignments
        cur_conjuncts = conjuncts + cur_assignments

        antecedent = ""
        if cur_conjuncts:
            antecedent += conjunct_assignments(cur_conjuncts)
        """
        if cur_conjuncts and next_conjuncts:
            antecedent += "&"
        if next_conjuncts:
            antecedent += f"next({conjunct_assignments(next_conjuncts)})"
        """
        amended_disjuncts.append(antecedent)
    antecedent_total = disjunct_assignments(amended_disjuncts)
    output = f"{antecedent_total}->{consequent}"
    # This is in case there was no antecedent to start with:
    output = re.sub(r"\(\s*&", "(", output)
    output = re.sub(r"\(\s*\|", "(", output)
    output = re.sub(r"\(\s*->", "(", output)
    output = re.sub(r"->\s*\|", "->", output)
    if assignments == [] and FASTLAS:
        return '\n'
    return f"\t{op}({output});\n"


def integrate_consequent(temp_op: str, assignments: list[str], antecedent: Optional[str], consequent: str):
    # next_assignments = [x for i, x in enumerate(assignments) if re.search("next", facts[i])]
    # ev_assignments = [x for i, x in enumerate(assignments) if re.search("eventually", facts[i])]
    # cur_assignments = [x for x in assignments if x not in ev_assignments and x not in next_assignments]
    cur = conjunct_assignments(assignments)
    # next = conjunct_assignments(next_assignments)
    # ev = conjunct_assignments(ev_assignments)
    """
    # This is for pRespondsToS patterns:
    respond = re.search(r"F\(([^)]*)\)", consequent)
    if respond:
        ev = f"F({disjunct_assignments([ev, respond.group(1)])})"
        consequent = consequent.replace(f"F({respond.group(1)})", '', 1)
    elif re.search(r"GF\(", antecedent):
        ev_old = consequent.replace(");", "")
        ev = disjunct_assignments([ev, ev_old])
        consequent = consequent.replace(ev_old, '', 1)
    # This is for next patterns:
    respond = re.search(r"next\(([^)]*)\)", consequent)
    if respond:
        next = f"next({disjunct_assignments([next, respond.group(1)])})"
        consequent = consequent.replace(f"next({respond.group(1)})", '', 1)
    elif next:
        next = f"next({next})"
    """
    cur = disjunct_assignments([cur] + get_disjuncts(consequent))
    # consequent = disjunct_assignments([cur, next, ev])
    consequent = disjunct_assignments([cur])
    if antecedent:
        output = f"\tG({antecedent}->{consequent});"
    else:
        output = f"\tG({consequent});"
    if assignments == [] and FASTLAS:
        return '\n'
    return f"{output}\n"


def extract_assignments_from_facts(facts, flip: bool):
    assignments = []
    for fact in facts:
        fact = fact.strip()
        is_negation = HoldsAtAtom.pattern.match(fact).group(HoldsAtAtom.NEG_PREFIX)
        atom = HoldsAtAtom.pattern.match(fact).group(HoldsAtAtom.ATOM).strip()
        value: bool = not bool(is_negation)
        value = not value if flip else value
        atom_assignment = f"{atom}={str(value).lower()}"

        assignments.append(atom_assignment)
    return assignments


def get_conjuncts(disjunct: str):
    return disjunct.split("&")


def get_disjuncts(conjunct: str):
    return conjunct.split("|")


def conjunct_assignments(assignments):
    assignments = [assignment for assignment in assignments if assignment != ""]
    output = '&'.join(assignments)
    return output


def disjunct_assignments(assignments):
    assignments = [assignment for assignment in assignments if assignment != ""]
    output = '|'.join(assignments)
    return output


def log_to_asp_trace(lines: str, trace_name: str = "trace_name_0") -> str:
    """
    Converts a runtime log into a workable trace string
    i.e.
    ->
    :param lines: Lines from log file
    :param trace_name: Name of Log
    :return: Trace string
    """
    ret = ""
    for i, line in enumerate(lines.split("\n")):
        ret += log_line_to_asp_trace(line, i, trace_name)
        ret += "\n"
    return ret


def log_line_to_asp_trace(line: str, idx: int = 0, trace_name: str = "trace_name_0") -> str:
    """
    Converts one line from a runtime log into a workable trace string
    i.e.
    ->
    :param line: <highwater:false, methane:false, pump:false, PREV_aux_0:false, Zn:0>
    :param idx: index where log line resides
    :param trace_name:
    :return:     not_holds_at(current,highwater,idx,trace_name).
                 not_holds_at(current,methane,idx,trace_name).
                 not_holds_at(current,pump,idx0,trace_name).
    """
    pairs = extract_string_boolean_pairs(line)
    filtered_pairs = [(key, value == 'true') for key, value in pairs if not key.startswith(('PREV', 'NEXT', 'Zn'))]
    ret = ""
    for env_var, is_true in filtered_pairs:
        ret += f"{'' if is_true else 'not_'}holds_at(current,{env_var},{idx},{trace_name}).\n"

    return ret


def extract_string_boolean_pairs(line):
    """
    Get all pairs of strings and booleans of form 'name:val'
    :param line:
    :return:
    """
    pattern = r"\b([a-zA-Z_][\w]*):(\btrue\b|\bfalse\b)"
    pairs = re.findall(pattern, line)
    return pairs


def spectra_to_df(spec: List[str]) -> pd.DataFrame:
    """
    Converts formatted Spectra file into Pandas DataFrame for manipulation into ASP.

    :param spec: Spectra specification as List of Strings.
    :return: Pandas DataFrame containing GR(1) expressions converted into antecedent/consequent.
    """
    formula_list = []
    for i, line in enumerate(spec):
        words = line.split(" ")
        if line.find('--') >= 0:
            name = re.sub(r":|\s", "", words[2])
            formula = re.sub('\s*', '', spec[i + 1])

            pRespondsToS, when = gr1_type_of(formula)

            formula_parts = formula.replace(");", "").split("->")
            if len(formula_parts) == 1:
                antecedent = ""
                consequent = re.sub(r"[^\(]*\(", "", formula_parts[0], 1)
            else:
                antecedent = re.sub(r"[^\(]*\(", "", formula_parts[0], 1)
                consequent = formula_parts[1]
            if pRespondsToS:
                consequent = re.sub(r"^F\(", "", consequent)

            formula_list.append(
                [words[0], name, formula,
                 antecedent,
                 consequent, when]
            )
    columns_and_types = OrderedDict([
        ('type', str),
        ('name', str),
        ('formula', str),
        ('antecedent', object),  # list[str]
        ('consequent', object),  # list[str]
        ('when', object)  # When
    ])
    spec_df = pd.DataFrame(formula_list, columns=list(columns_and_types.keys()))
    # Set the data types for each column
    for col, dtype in columns_and_types.items():
        spec_df[col] = spec_df[col].astype(dtype)

    return spec_df


def gr1_type_of(formula):
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


def filter_formulas_of_type(formula_df: pd.DataFrame, expression: ExpType) -> pd.DataFrame:
    return formula_df.loc[formula_df['type'] == str(expression)]


def parse_formula_str(formula: str) -> List[Dict[str, List[str]]]:
    """
    Parse a formula consisting of disjunctions and conjunctions of temporal operators.

    Args:
        formula (str): The input formula to parse.

    Returns:
        List[Dict[str, List[str]]]: A list of dictionaries containing operators and their associated literals.
    """
    # Remove any whitespace for easier processing
    formula = formula.replace(" ", "")

    # Split the formula by disjunction (e.g., '|' or '∨')
    disjunctions = formula.split('|')
    parsed_conjunctions = []

    for conjunct in disjunctions:
        conjunct = remove_outer_parentheses(conjunct)
        conjunct_dict = defaultdict(list)

        # Split each conjunct by conjunctions (e.g., '&')
        parts = split_with_outer_parentheses(conjunct)

        for part in parts:
            # Regex to capture "operator(content)"
            match = re.match(r'^(next|prev|PREV|G|F)\((.+)\)', part)

            if match:
                operator = match.group(1)
                operator = "eventually" if operator == "F" else operator
                operator = "prev" if operator == "PREV" else operator
                content = match.group(2)
                # Split content by '&' and add to corresponding operator
                literals = re.split(r'\s*&\s*', content)
                conjunct_dict[operator].extend(literals)
            else:
                # No temporal operator, assume 'current' (no operation)
                literals = re.split(r'\s*&\s*', part.strip("()"))
                conjunct_dict["current"].extend(literals)

        parsed_conjunctions.append(dict(conjunct_dict))

    return parsed_conjunctions


def split_with_outer_parentheses(input_str: str) -> List[str]:
    """
    Split the input string based on operators while considering outer parentheses.

    Args:
        input_str (str): The input string to split.

    Returns:
        List[str]: A list of segments split based on the defined logic.
    """
    # This regex captures '&' not enclosed within parentheses
    pattern = r'\b(next|prev|PREV|F|G)\(([^()]*|[^&]*)*\)|[^()&\s]+'
    segments = [match.group(0) for match in re.finditer(pattern, input_str)]

    # Clean up the segments and filter out empty strings
    return [segment.strip() for segment in segments if segment.strip()]


def remove_outer_parentheses(s):
    s = s.strip()
    # Check if the string starts and ends with parentheses
    if s.startswith('(') and s.endswith(')'):
        return s[1:-1]  # Remove the first and last character
    return s  # Return the original string if conditions are not met


def generate_trace_asp(strong_spec_file, ideal_spec_file, trace_file):
    try:
        old_trace = read_file_lines(trace_file)
    except FileNotFoundError:
        old_trace = []
    asp_restrictions = compose_old_traces(old_trace)

    trace = {}

    initial_expressions, prevs, primed_expressions, unprimed_expressions, variables \
        = extract_expressions_from_file(ideal_spec_file, counter_strat=True)
    initial_expressions_s, prevs_s, primed_expressions_s, unprimed_expressions_s, variables_s \
        = extract_expressions_from_file(strong_spec_file, counter_strat=True)

    # To include starting guarantees:
    ie_g, prevs_g, pe_g, upe_g, v_g = extract_expressions_from_file(strong_spec_file, guarantee_only=True)
    initial_expressions += ie_g
    primed_expressions += pe_g
    unprimed_expressions += upe_g

    # initial_expressions_sa, prevs_sa, primed_expressions_sa, unprimed_expressions_sa, variables_sa = extract_expressions(
    #     start_file, counter_strat=True)

    # This adds starting guarantees to final assumptions
    # initial_expressions += [x for x in initial_expressions_s if x not in initial_expressions_sa]
    # primed_expressions += [x for x in primed_expressions_s if x not in primed_expressions_sa]
    # unprimed_expressions += [x for x in unprimed_expressions_s if x not in unprimed_expressions_sa]

    expressions = primed_expressions + unprimed_expressions
    neg_expressions = primed_expressions_s + unprimed_expressions_s

    variables = [var for var in variables if not re.search("prev|next", var)]

    # Lowercasing PREV in expressions
    expressions = [re.sub(r"PREV\((!*)([^\|^\(]*)\)", r"\1prev_\2", x) for x in expressions]
    neg_expressions = [re.sub(r"PREV\((!*)([^\|^\(]*)\)", r"\1prev_\2", x) for x in neg_expressions]
    # Removing braces around next function args (`next(sth)` -> `next_sth`)
    expressions = [re.sub(r"next\((!*)([^\|^\(]*)\)", r"\1next_\2", x) for x in expressions]
    neg_expressions = [re.sub(r"next\((!*)([^\|^\(]*)\)", r"\1next_\2", x) for x in neg_expressions]

    one_point_exp = [re.sub(r"(" + '|'.join(variables) + r")", r"prev_\1", x) for x in
                     unprimed_expressions + initial_expressions]
    expressions += one_point_exp
    expressions += [re.sub(r"(" + '|'.join(variables) + r")", r"next_\1", x) for x in unprimed_expressions]
    neg_one_point_exp = [re.sub(r"(" + '|'.join(variables) + r")", r"prev_\1", x) for x in
                         unprimed_expressions_s + initial_expressions_s]
    neg_expressions += neg_one_point_exp
    neg_expressions += [re.sub(r"(" + '|'.join(variables) + r")", r"next_\1", x) for x in unprimed_expressions_s]

    expressions += two_period_primed_expressions(primed_expressions, variables)
    neg_expressions += two_period_primed_expressions(primed_expressions_s, variables)

    # Can it be done with one time point?
    state, violation = generate_model(one_point_exp,
                                      neg_one_point_exp,
                                      variables, scratch=True,
                                      asp_restrictions=asp_restrictions)
    if state is not None and len(neg_one_point_exp) > 0:
        trace[0] = [re.sub(r"prev_", "", var) for var in state[0] if re.search("prev_", var)]
        write_trace(trace, trace_file)
        return trace_file, violation

    # Can it be done with two time points?
    two_point_exp = [x for x in expressions if not re.search("next", x)]
    two_point_neg_exp = [x for x in neg_expressions if not re.search("next", x)]
    state, violation = generate_model(two_point_exp,
                                      two_point_neg_exp, variables, scratch=True,
                                      asp_restrictions=asp_restrictions)
    if state is not None and len(two_point_neg_exp) > 0:
        trace[0] = [re.sub(r"prev_", "", var) for var in state[0] if re.search("prev_", var)]
        trace[1] = [var for var in state[0] if not re.search("prev_|next_", var)]
        write_trace(trace, trace_file)
        return trace_file, violation

    # Can it be done with three time points?
    state, violation = generate_model(expressions, neg_expressions, variables, scratch=True,
                                      asp_restrictions=asp_restrictions)
    if state is None or len(neg_expressions) == 0:
        return None, None
    trace[0] = [re.sub(r"prev_", "", var) for var in state[0] if re.search("prev_", var)]
    trace[1] = [var for var in state[0] if not re.search("prev_|next_", var)]
    trace[2] = [re.sub(r"next_", "", var) for var in state[0] if re.search("next_", var)]
    write_trace(trace, trace_file)
    return trace_file, violation


def write_trace(trace, filename):
    try:
        prev = read_file_lines(filename)
        timepoint = int(max(re.findall(r"trace_name_(\d*)", ''.join(prev)))) + 1
    except FileNotFoundError:
        timepoint = 0
    trace_name = "trace_name_" + str(timepoint)
    output = ""
    for timepoint in trace.keys():
        variables = list(trace[timepoint])
        for var in variables:
            if not re.search(r"prev_", var):
                prefix = ""
                if var[0] == "!":
                    prefix = "not_"
                    var = var[1:]
                output += prefix + "holds_at(" + var + "," + str(timepoint) + "," + trace_name + ").\n"
        output += "\n"
    with open(filename, 'a', newline='\n') as file:
        file.write(output)


def compose_old_traces(old_trace):
    if old_trace == []:
        return ""
    string = ''.join(old_trace)
    traces = re.findall(r"trace_name_\d*", string)
    traces = list(dict.fromkeys(traces))
    output = "\n"
    for i, name in enumerate(traces):
        assignments = []
        for n in range(3):
            as_name = "as" + str(i) + "_" + str(n)
            assignments += asp_trace_to_spectra(name, string, n)
            output += as_name + " :- " + ','.join(assignments) + ".\n"
            output += ":- " + as_name + ".\n"
    return output


def two_period_primed_expressions(primed_expressions, variables):
    nexts = [x for x in primed_expressions if not re.search("PREV|prev", x)]
    prevs = [x for x in primed_expressions if not re.search("next", x)]
    next2_3 = [re.sub(r"next\((!*)([^\|^\(]*)\)", r"\1next_\2", x) for x in nexts]
    next1_2 = [re.sub("(" + "|".join(variables) + ")", r"prev_\1", x) for x in nexts]
    next1_2 = [re.sub(r"next\((!*)([^\|^\(]*)\)", r"\1next_\2", x) for x in next1_2]
    next1_2 = [re.sub(r"next_prev_", "", x) for x in next1_2]

    prev1_2 = [re.sub(r"PREV\((!*)([^\|^\(]*)\)", r"\1prev_\2", x) for x in prevs]
    prev2_3 = [re.sub("(" + "|".join(variables) + ")", r"next_\1", x) for x in prevs]
    prev2_3 = [re.sub(r"PREV\((!*)([^\|^\(]*)\)", r"\1prev_\2", x) for x in prev2_3]
    prev2_3 = [re.sub(r"prev_next_", "", x) for x in prev2_3]
    return next1_2 + next2_3 + prev1_2 + prev2_3


def extract_expressions_from_file(file, counter_strat=False, guarantee_only=False):
    spec = read_file_lines(file)
    return extract_expressions_from_spec(spec, counter_strat, guarantee_only)


def extract_expressions_from_spec(spec: list[str], counter_strat=False, guarantee_only=False):
    variables = strip_vars(spec)
    spec = simplify_assignments(spec, variables)
    assumptions = extract_non_liveness(spec, "assumption")
    guarantees = extract_non_liveness(spec, "guarantee")
    if counter_strat:
        guarantees = []
    if guarantee_only:
        assumptions = []
    prev_expressions = [re.search(r"G\((.*)\);", x).group(1) for x in assumptions + guarantees if
                        re.search(r"PREV", x) and re.search("G", x)]
    list_of_prevs = [f"PREV\\({s}\\)" for s in variables + [f"!{x}" for x in variables]]
    prev_occurances = [re.findall('|'.join(list_of_prevs), exp) for exp in prev_expressions]
    prevs = [item for sublist in prev_occurances for item in sublist]
    prevs = [re.sub(r"PREV\(!*(.*)\)", r"prev_\1", x) for x in prevs]
    prevs = list(dict.fromkeys(prevs))
    variables += prevs
    variables.sort()

    unprimed_expressions = [re.search(r"G\(([^F]*)\);", x).group(1) for x in assumptions + guarantees if
                            not re.search(r"PREV|next", x) and re.search(r"G\s*\(", x)]
    primed_expressions = [re.search(r"G\(([^F]*)\);", x).group(1) for x in assumptions + guarantees if
                          re.search(r"PREV|next", x) and re.search("G", x)]
    initial_expressions = [x.strip(";") for x in assumptions + guarantees if not re.search(r"G\(|GF\(", x)]
    return initial_expressions, prevs, primed_expressions, unprimed_expressions, variables


def extract_non_liveness(spec, exp_type):
    output = extract_all_expressions(exp_type, spec)
    return [spectra_to_DNF(x) for x in output if not re.search("F", x)]


def generate_model(expressions, neg_expressions, variables, scratch=False, asp_restrictions="", force=False):
    if scratch:
        prevs = ["prev_" + var for var in variables]
        nexts = ["next_" + var for var in variables]
        if any([re.search("next", x) for x in expressions + neg_expressions]):
            variables = variables + prevs + nexts
        # TODO: double check regex, ensure it's correct
        elif any([re.search(r"\b" + r"|\b".join(variables), x) for x in expressions + neg_expressions]):
            variables = variables + prevs
        else:
            variables = prevs
        output = asp_restrictions + "\n"
    else:
        output = ""
    expressions = aspify(expressions)
    for i, rule in enumerate(expressions):
        name = f"t{i}"
        disjuncts = [x.strip() for x in rule.split(";")]
        for disjunct in disjuncts:
            output += f"{name} :- {disjunct}.\n"
        output += f"s{name} :- not {name}.\n"
        output += f":- s{name}.\n"

    for variable in variables:
        output += f"{{{variable}}}.\n"

    neg_expressions = aspify(neg_expressions)
    rules = []
    for i, rule in enumerate(neg_expressions):
        name = f"rule{i}"
        disjuncts = [x.strip() for x in rule.split(";")]
        for disjunct in disjuncts:
            output += f"{name} :- {disjunct}.\n"
        rules.append(name)

    if len(rules) > 0:
        output += f":- {','.join(rules)}.\n"
    for var in variables:
        output += f"#show {var}/0.\n"

    file = "/tmp/temp_asp.lp"
    write_file(file, output)
    clingo_out = run_clingo_raw(file, n_models=0)

    violation = True

    matches = re.findall(r'Answer:\s*\d+(?:.*)?\r?\n([^\r\n]*)', clingo_out)

    if not matches:
        # print(clingo_out)
        # print("Something not right with model generation")
        return None, None
    states = [match.split() for match in matches]
    for state in states:
        [state.append(f"!{x}") for x in variables if x not in state]
    return states, violation


def asp_trace_to_spectra(name, string, n):
    tups = re.findall(r"\b(.*)holds_at\((.*)," + str(n) + "," + name + r"\)", string)
    prefix = ""
    if n == 2:
        prefix = "next_"
    if n == 0:
        prefix = "prev_"
    output = ["not " + prefix + tup[1] if tup[0] == "not_" else prefix + tup[1] for tup in tups]
    return output


def simplify_assignments(spec, variables):
    vars = "|".join(variables)
    spec = [re.sub(rf"({vars})=true", r"\1", line) for line in spec]
    spec = [re.sub(rf"({vars})=false", r"!\1", line) for line in spec]
    return spec


def extract_all_expressions(exp_type, spec):
    search_type = exp_type
    if exp_type in ["asm", "assumption"]:
        search_type = "asm|assumption"
    if exp_type in ["gar", "guarantee"]:
        search_type = "gar|guarantee"
    output = [re.sub(r"\s", "", spec[i + 1]) for i, line in enumerate(spec) if re.search(search_type, line)]
    return output


def spectra_to_DNF(formula):
    prefix = ""
    suffix = ";"
    justice = re.search(r"G\((.*)\);", formula)
    liveness = re.search(r"GF\((.*)\);", formula)
    if justice:
        prefix = "G("
        suffix = ");"
        pattern = justice
    if liveness:
        prefix = "GF("
        suffix = ");"
        pattern = liveness
    if not justice and not liveness:
        non_temporal_formula = formula
    else:
        non_temporal_formula = pattern.group(1)
    parts = non_temporal_formula.split("->")
    if len(parts) == 1:
        return prefix + non_temporal_formula + suffix
    return prefix + '|'.join([negate(parts[0]), parts[1]]) + suffix


def aspify(expressions):
    # is this first one ok?
    expressions = [re.sub(r"\(|\)", "", x) for x in expressions]
    expressions = [re.sub(r"\|", ";", x) for x in expressions]
    expressions = [re.sub(r"!", " not ", x) for x in expressions]
    expressions = [re.sub(r"&", ",", x) for x in expressions]
    return expressions


def run_clingo_raw(filename, n_models: int = 1) -> str:
    filepath = f"{filename}"
    cmd = create_cmd(['clingo', f'--models={n_models}', filepath])
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
    return output.decode('utf-8')


def run_all_unrealisable_cores(spectra_str: str) -> List[Set[str]]:
    """
    Gets the names of all unrealisable cores from a given spectra specification as string.
    """
    temp_spectra_file = generate_temp_filename(ext=".spectra")
    write_to_file(temp_spectra_file, spectra_str)
    pRespondsToS_substitution(temp_spectra_file)
    output = run_all_unrealisable_cores_raw(temp_spectra_file)
    core_nums_list: List[Set[int]] = _extract_cores(output)
    core_names_list = []
    for core_nums in core_nums_list:
        core_names = set()
        for core_num in core_nums:
            line_with_name = get_line_from_file(temp_spectra_file, core_num)
            name = line_with_name.split("--")[1].strip()
            core_names.add(name)
        core_names_list.append(core_names)
    return core_names_list


def _extract_cores(text) -> List[Set[int]]:
    # Split to get the part after "Final results:"
    parts = text.split("Final results:")
    if len(parts) < 2:
        return []

    final_section = parts[1]

    # Pattern to match lines like "Core #1 at lines < 12 15 >"
    pattern = re.compile(r'Core\s+#\d+\s+at\s+lines\s+<\s*([\d\s]*)\s*>')

    results = []
    for match in pattern.finditer(final_section):
        numbers = match.group(1).strip()
        if numbers:  # Only add non-empty sets
            num_set = {int(n) for n in numbers.split()}
            results.append(num_set)

    return results


def run_all_unrealisable_cores_raw(filename) -> str:
    filepath = f"{filename}"
    args = jpype.JArray(JString)([filepath, "--jtlv"])
    output = SpectraToolbox.exploreAllCores(args)
    return str(output)


def shift_prev_to_next(formula, variables):
    # Assumes no nesting of next/prev
    # filt = r'PREV\(' + r'|PREV\('.join(variables) + r'|PREV\(!'.join(variables)
    filt = "PREV"
    if not re.search(filt, formula):
        return re.sub("next", "X", formula)
    formula = re.sub("next", "XX", formula)

    all_vars = '|'.join(["!" + var + "|" + var for var in variables])
    # formula = re.sub(r"([^\(^!])(" + all_vars + r")|([^V^X])\((" + all_vars + ")", r"\1X(\2)", formula)
    formula = re.sub(f"([^V^X])\(({all_vars})", r"\1(X(\2)", formula)
    formula = re.sub(f"([^\(^!])({all_vars})", r"\1X(\2)", formula)

    formula = re.sub(r"PREV\((" + all_vars + r")\)", r"\1", formula)
    return formula
    # save this as explanation of above:
    # re.sub(r"([^\(^!])(!highwater|highwater|!pump|pump)|([^V^X])\((!highwater|highwater|!pump|pump)", r"\1X(\2)", formula)
    # use this to test:
    # temp_formula ='G(PREV(pump)&PREV(!methane)&!highwater&methane&!methane&pump->XX(!highwater)&XX(methane));'

    # re.sub(r"([^V^X])\((!pump)", r"\1(X(\2))", formula)


def semantically_identical_spot(to_cmp_file, baseline_file):
    to_cmp_file = re.sub("_patterned\.spectra", ".spectra", to_cmp_file)
    assumption = equivalent_expressions("assumption|asm", to_cmp_file, baseline_file)
    if assumption is None:
        return SimEnv.Invalid
    if not assumption:
        if realizable(to_cmp_file):
            return SimEnv.Realizable
        else:
            # This should never happen:
            return SimEnv.Unrealizable
    guarantee = equivalent_expressions("guarantee|gar", to_cmp_file, baseline_file)
    if guarantee is None:
        print("Guarantees Not Working in Spot:\n" + to_cmp_file)
    if not guarantee:
        return SimEnv.IncorrectGuarantees
    return SimEnv.Success


def equivalent_expressions(exp_type, start_file, end_file):
    start_exp = extract_all_expressions_spot(exp_type, start_file)
    end_exp = extract_all_expressions_spot(exp_type, end_file)
    linux_cmd = ["ltlfilt", "-c", "-f", f"{start_exp}", "--equivalent-to", f"{end_exp}"]
    p = subprocess.Popen(linux_cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    output = p.communicate()[0]
    close_file_descriptors_of_subprocess(p)
    output = output.decode('utf-8')
    reg = re.search(r"(\d)\n", output)
    if not reg:
        return None
    result = reg.group(1)
    if result == "0":
        return False
    if result == "1":
        return True
    return None


def close_file_descriptors_of_subprocess(p):
    if p.stdin:
        p.stdin.close()
    if p.stdout:
        p.stdout.close()
    if p.stderr:
        p.stderr.close()


def extract_all_expressions_spot(exp_type, file, return_list=False):
    spec = read_file_lines(file)
    variables = strip_vars(spec)
    spec = simplify_assignments(spec, variables)
    expressions = [re.sub(r"\s", "", spec[i + 1]) for i, line in enumerate(spec) if re.search("^" + exp_type, line)]
    expressions = [shift_prev_to_next(formula, variables) for formula in expressions]
    if any([re.search("PREV", x) for x in expressions]):
        raise Exception("There are still PREVs in the expressions!")
    if return_list:
        return [re.sub(";", "", x) for x in expressions]
    exp_conj = re.sub(";", "", '&'.join(expressions))
    return exp_conj


def violations_in_initial_conditions(file):
    '''
    This is because the Spectra CLI is inconsistent in throwing errors relating to initial conditions. Initial
    conditions cannot refer to primed (next) variables. Initial assumptions cannot refer to system variables.
    :param file:
    :return:
    '''
    spec = read_file_lines(file)
    sys_vars = strip_vars(spec, "sys")
    inits = [(line, spec[i + 1]) for i, line in enumerate(spec) if
             line.find("--") >= 0 and not re.search(r"G|F|pRespondsToS", spec[i + 1])]
    if any([bool(re.search(r"next|X", tup[1])) for tup in inits]):
        print("Initial expression contains primed (next) variables.")
        return True
    sys_vars = [re.escape(var) for var in sys_vars]
    init_ass = [tup[1] for tup in inits if re.search(r"assumption", tup[0])]
    if any([re.search(r'|'.join(sys_vars), ass) for ass in init_ass]):
        print("Initial assumption refers to system variables.")
        return True
    return False


def realizable(file, suppress=False):
    if violations_in_initial_conditions(file):
        print("Spectra file in wrong format for CLI realizability check: (initial conditions)")
        print(file)
        return None
    file = pRespondsToS_substitution(file)
    args = ["-i", file, "--jtlv"]
    output = run_spectra_cli(args)
    if re.search("Result: Specification is unrealizable", output):
        return False
    elif re.search("Result: Specification is realizable", output):
        return True
    if not suppress:
        print(output)
    print("Spectra file in wrong format for CLI realizability check:")
    print(file)
    return None

def synthesise_extract_counter_strategies(file):
    if violations_in_initial_conditions(file):
        print("Spectra file in wrong format for CLI realizability check: (initial conditions)")
        print(file)
        return None
    file = pRespondsToS_substitution(file)
    args = ["-i", file, "--counter-strategy", "--jtlv"]
    output = run_spectra_cli(args)
    return output

def synthesise_controller(spec_file_path, output_folder_path, suppress=False) -> bool:
    if violations_in_initial_conditions(spec_file_path):
        print("Spectra file in wrong format for CLI realizability check: (initial conditions)")
        print(spec_file_path)
        return False
    # Check if parent directory exists
    parent_dir = os.path.dirname(output_folder_path)
    if not os.path.exists(parent_dir):
        print(f"Error: Path to output folder does not exist: {parent_dir}")
        return False

    spec_file_path = pRespondsToS_substitution(spec_file_path)
    args = ["-i", spec_file_path, "--jtlv", '-s', '--static', '-o', output_folder_path]
    output = run_spectra_cli(args)
    if re.search("Error: Cannot synthesize an unrealizable specification", output):
        print("Error: Cannot synthesize an unrealizable specification")
        return False
    elif re.search("Result: Specification is realizable", output):
        return True
    if not suppress:
        print(output)
    print("Spectra file in wrong format for CLI realizability check:")
    print(spec_file_path)
    return False

def run_spectra_cli(args: list[str]) -> str:
    """
    Run a Java main method and capture its printed output as a string.

    Parameters:
    - args: list of string arguments to pass to main()

    Returns:
    - Captured standard output as a Python string.
    """
    if not jpype.isJVMStarted():
        raise RuntimeError("JVM is not started. Start it with jpype.startJVM() before calling this function.")

    # Import Java system classes
    java_lang_System = jpype.JPackage("java.lang").System
    java_io_ByteArrayOutputStream = jpype.JPackage("java.io").ByteArrayOutputStream
    java_io_PrintStream = jpype.JPackage("java.io").PrintStream

    # Backup original System.out
    original_out = java_lang_System.out

    # Prepare streams to capture output
    baos = java_io_ByteArrayOutputStream()
    ps = java_io_PrintStream(baos)

    # Redirect System.out to our PrintStream
    java_lang_System.setOut(ps)

    try:
        # Load the Java class and convert args to Java String[]
        java_args = JArray(JString)(args)

        # Call the main method
        SpectraCLI.main(java_args)

        # Flush and get captured output as bytes
        ps.flush()
        output_bytes = baos.toByteArray()

        # Decode bytes to Python string
        output_str = bytes(output_bytes).decode("utf-8")

    finally:
        # Restore original System.out no matter what
        java_lang_System.setOut(original_out)

    return output_str

def remove_double_outer_brackets(string):
    if string[0:2] == "((" and string[-3:-1] == "))":
        return string[1:-1]
    return string


def negate(string):
    '''
    Assumes precedence of AND (DNF)
    :param string:
    :return:
    '''
    # examples:
    # string1 = 'F(level_1_nest_0)|F(level_1_nest_1)|F(level_1_nest_2)'
    # string2 = "A|B&C"
    # string = "(level_1)W(level_2)"
    if string == "":
        return string
    disjuncts = re.sub(r"\s", "", string).split("|")
    for i, sub_string in enumerate(disjuncts):
        conjuncts = sub_string.split("&")
        conjuncts = ["!" + x for x in conjuncts]
        conjuncts = push_negations(conjuncts)
        # This way we push F's out if they are common
        conjunct = check_first_chars(conjuncts, "conjuncts")
        # conjunct = "|".join(conjuncts)
        if len(conjuncts) > 1 and len(disjuncts) > 1:
            conjunct = "(" + conjunct + ")"
        conjunct = remove_double_outer_brackets(conjunct)
        disjuncts[i] = conjunct
    disjuncts = push_negations(disjuncts)
    # This is if we want to push G's out, which i've decided we don't
    # disjuncts = check_first_chars(disjuncts, "disjuncts")
    # return disjuncts
    output = '&'.join(disjuncts)
    output = remove_trivial_outer_brackets(output)
    return output


def check_first_chars(list, type):
    if len(list) == 1:
        return list[0]
    if type == "conjuncts":
        dist_char = "F"
        join_char = "|"
    if type == "disjuncts":
        dist_char = "G"
        join_char = "&"

    first_chars = [chars[0:2] for chars in list]
    character = first_chars[0]
    if all(character == char for char in first_chars):
        if character in ["X(", dist_char + "("]:
            list = [chars[2:-1] for chars in list]
            output = character[0] + "(" + join_char.join(list) + ")"
            return output
    output = join_char.join(list)
    return output


def push_negations(disjuncts):
    disjuncts = [re.sub(r"!\((.*)\)W\((.*)\)", r"(!\2)U((!\2)&(!\1))", x) for x in disjuncts]
    disjuncts = [re.sub(r"!\((.*)\)U\((.*)\)", r"(!\2)W((!\2)&(!\1))", x) for x in disjuncts]
    disjuncts = [re.sub(r"!!", r"", x) for x in disjuncts]
    disjuncts = [re.sub(r"!F\(", r"G(!", x) for x in disjuncts]
    disjuncts = [re.sub(r"!G\(", r"F(!", x) for x in disjuncts]
    disjuncts = [re.sub(r"!X\(", r"X(!", x) for x in disjuncts]
    disjuncts = [re.sub(r"!next\(", r"next(!", x) for x in disjuncts]
    disjuncts = [re.sub(r"!PREV\(", r"PREV(!", x) for x in disjuncts]
    disjuncts = [re.sub(r"!\(", r"(!", x) for x in disjuncts]
    disjuncts = [re.sub(r"!!", r"", x) for x in disjuncts]
    return disjuncts


def split_expression_to_raw_components(exp: str) -> List[str]:
    exp_components: List[str] = exp.split("->")
    if len(exp_components) == 1:
        exp = re.sub(r"(G|GF)\(\s*", r"\1(true -> ", exp_components[0])
        exp_components = exp.split("->")
    exp_components = [comp.strip() for comp in exp_components]
    return exp_components


def eventualise_consequent(exp, learning_type: Learning):
    match learning_type:
        case Learning.ASSUMPTION_WEAKENING:
            line = split_expression_to_raw_components(exp)
            return eventualise_consequent_assumption(line)
        case Learning.GUARANTEE_WEAKENING:
            line = split_expression_to_raw_components(exp)
            return eventualise_consequent_assumption(line)
            raise NotImplemented(
                "Not sure yet if we want to weaken guarantees by introducing eventually to their consequent.")
        case _:
            raise ValueError("No such learning type")


def extract_contents_of_temporal(expression: str):
    # Remove "next", "prev", or "X" (case-insensitive) and surrounding parentheses
    return re.sub(r'(?i)(next|prev|X)\s*\(([^)]*)\)|\)$', r'\2', expression)


def eventualise_consequent_assumption(line: List[str]):
    antecedent = line[0]
    consequent = line[1]
    consequent_without_temporal = extract_contents_of_temporal(consequent)
    ev_consequent = re.sub(r'^(.*?)(;)?$', r'F(\1)\2', consequent_without_temporal)
    output = antecedent + "->" + ev_consequent
    return '\t' + output + "\n"


def re_line_spec(spec: list[str]) -> list[str]:
    """
    Move multiple newlines to new elems in list.
    Ensures every separate elem has a newline at the end.
    e.g.: ["Anna\n\n", "\n", "eats\n", "potatoes"]
        -> ["Anna\n", "\n", "\n", "eats\n", "potatoes\n"]
    :param spec: Specification as list of strings
    :return: new_spec: Specification reformatted as explained above
    """
    return [line + '\n' for line in ''.join(spec).split("\n")]


def shutdown():
    def force_exit():
        print("Shutdown taking too long, forcing exit.")
        os._exit(1)

    print("Shutting down SpectraTool and SpectraCLI...")
    timer = threading.Timer(10, force_exit)
    timer.start()

    # SpectraToolbox.shutdownNow()
    jpype.shutdownJVM()

    print("JVM shutdown initiated...")
    timer.cancel()
    print("JVM shutdown complete.")


atexit.register(shutdown)
