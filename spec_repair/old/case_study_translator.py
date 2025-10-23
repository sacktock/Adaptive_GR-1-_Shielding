import csv
import os
import copy
import re

from spec_repair.util.spec_util import pRespondsToS_substitution, parenthetic_contents, \
    realizable, negate, push_negations
from spec_repair.old.specification_helper import get_name
from spec_repair.util.file_util import read_file_lines, write_file


def parenthetic_contents_with_function(string, include_after=False):
    """Generate parenthesized contents in string as pairs (level, contents)."""
    stack = []
    for i, c in enumerate(string):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            pre = string[start - 1:start]
            # if start == 0:
            #     pre = string[0]
            if include_after:
                if i + 2 > len(string):
                    post = ""
                else:
                    post = string[i + 1:i + 2]
                yield (len(stack), pre, string[start + 1: i], post)
            else:
                yield (len(stack), pre, string[start + 1: i])


def push_negations_non_DNF(formula):
    # formula = 'G(!(((X(g1)))&((X(g2)))))'

    formula_mapping, negated_formula_mapping, nest_names = express_formula_as_levels(formula)

    for key in formula_mapping.keys():
        formula = re.sub(re.escape(key), push_negations([formula_mapping[key]])[0], formula)
        for neg_key in negated_formula_mapping.keys():
            name = "!" + neg_key
            formula = re.sub(re.escape(name), push_negations([formula_mapping[name]])[0], formula)
    return formula
    # return push_negations([formula])[0]


def strip_unnecessary_brackets(formula):
    '''
    Must still be in the form of G,F,X
    :param formula:
    :return:
    '''
    # formula = "G(((!r2)&(g2))|((r2)&(!g2))->((!r2)&((X(!r2))))|((r2)&((X(r2)))))"
    # formula = "G(!g_0|(!g_1))"
    # formula = negate("!" + formula)
    while True:
        start_len = len(formula)
        # formula = push_negations([formula])[0]
        contents = list(parenthetic_contents_with_function(formula, include_after=True))
        unary_ops = ["X", "G", "F", "!"]
        bin_ops = ["W", "U"]
        doubled = [tup[2] for tup in contents if tup[1] not in unary_ops + bin_ops and tup[3] not in bin_ops]
        # This is for cases like 'GF(((!m|!p)&X(!h)))':
        doubled = [x for x in doubled if x.find("|") < 0 or x.find("&") >= 0]
        for sub_form in doubled:
            pattern = r"([^W^U^X^G^F^\!])" + re.escape("(" + sub_form + ")")
            formula = re.sub(pattern, r"\1" + sub_form, formula)
        if len(formula) == start_len:
            return formula


def add_missing_brackets(formula):
    # formula = 'G(r1->Fg1)'
    reg = re.search("^(.*)G([^F]*)F(.*)", formula)
    if reg:
        return reg.group(1) + "G" + reg.group(2) + "F(" + reg.group(3) + ")"
    return formula


def remove_nested_next(formula):
    # Assumes only one formula i.e. not G(a->b)&G(c->d)
    # formula = "G(!g_0&true|true&!g_1&r_0&X(r_1)->X(X(g_0&g_1)));"
    contents = list(parenthetic_contents_with_function(formula))
    if not any([tup[1] == "X" and tup[2][0] == "X" for tup in contents]):
        return formula
    sub_formula = [tup[2] for tup in contents if tup[0] == 0][0]
    wrap = re.sub(re.escape(sub_formula), "temp_param", formula)
    string, names, subformulae = replace_brackets_with_names(sub_formula)
    parts = re.split(r"(&|\||->)", string)
    for i, part in enumerate(parts):
        if part[0] == "X":
            parts[i] = part[2:-1]
        elif part not in ["&", "|", "->"]:
            parts[i] = "PREV(" + part + ")"
    string = ''.join(parts)
    output = replace_names_with_formulae(subformulae, names, string)
    return remove_nested_next(re.sub("temp_param", output, wrap))


def contrapositive_antecedent_nexts(formula):
    # formula = 'G(X(cc)->ca&go)'
    formula = re.sub("next", "X", formula)
    contents = list(parenthetic_contents_with_function(formula))
    first_level = [tup[1:] for tup in contents if tup[0] == 0]
    out_list = []
    for i, tup in enumerate(first_level):
        parts = tup[1].split("->")
        if len(parts) == 1:
            output = tup[1]
        elif parts[0].find("X") >= 0 and parts[1].find("X") < 0:
            output = negate_and_simplify(parts[1]) + "->" + negate_and_simplify(parts[0])
        # If X's on both sides, we convert to DNF (so that we have no antecedent) actually not necessary as if there
        # is a next on both sides, both sides are trivially satisfied when no next timepoint, not causing an error.
        # only when the antecedent holds and the consequent doesn't hold does a violation occur. elif parts[0].find(
        # "X") >= 0 and parts[1].find("X") >= 0: output = negate_and_simplify(parts[0]) + "|" + parts[1]
        else:
            output = tup[1]
        out_list.append(tup[0] + "(" + output + ")")
    first_level_combined = ["(".join(tup) + ")" for tup in first_level]
    for i, sub_form in enumerate(out_list):
        formula = re.sub(re.escape(first_level_combined[i]), sub_form, formula)
    return re.sub("X", "next", formula)


def gr_one(formula, variables):
    # formula = re.sub(r"\s", "", "(((G (F (r_0))) && (G (F (r_1)))) <-> (G (F (g))))")
    # formula = "G((((!r2)&&(g2))||((r2)&&(!g2)))->(((!r2)&&((X(!r2))))||((r2)&&((X(r2))))))"
    # formula = "G(r1->Fg1)"
    # formula = "(F G !(p)) <-> (G F acc)"
    # formula = 'G((p&&X(p))->X(X(!h)))'
    # formula = 'G(!(((X(g1)))&&((X(g2)))))'
    # formula = 'GF((!m|!p)&X(!h))'
    for var in variables:
        for op in ["X", "F", "G"]:
            # This is for when case studies supply temporal operators with no brackets
            formula = re.sub(op + r"!" + var, op + "(!" + var + ")", formula)
            formula = re.sub(op + var, op + "(" + var + ")", formula)
    formula = re.sub(r"\|+", r"|", formula)
    formula = re.sub(r"&+", r"&", formula)
    formula = re.sub(r"\(\((.*)\)->\((.*)\)", r"(\1->\2", formula)
    formula = add_missing_brackets(formula)
    # formula = 'G(!(((X(g1)))&((X(g2)))))'
    # formula = 'G((!(g_0))|(!(g_1)))'
    formula = push_negations_non_DNF(formula)
    formula = strip_unnecessary_brackets(formula)
    # formula = negate_and_simplify("!" + formula)
    formula = simplify_liveness(formula)
    formula = remove_nested_next(formula)
    formula = contrapositive_antecedent_nexts(formula)
    # formula = re.sub(r"X", r"next", formula)

    # is this necessary?:
    while formula[0] == "(" and formula[-1] == ")":
        formula = formula[1:-1]
    return formula


def translate_spec_to_file(spec, filename, BC=""):
    name = get_name(filename)
    output_spec = "module " + re.sub(r"-", "_", name) + "\n\n"
    text = re.sub(r"\s", r"", ''.join(spec))
    text = re.sub(r"''-(.)=", r"'-\1='", text)
    elements = re.split(r"(-g=|-d=|-ins=|-outs=|-nbc=)", text)
    variables = []

    output_spec += extract_variables(elements, "-ins=", "env", variables)
    output_spec += extract_variables(elements, "-outs=", "sys", variables)
    output_spec += '\n'

    output_spec += extract_expressions(elements, "guarantee", "-g=", variables)
    output_spec += extract_expressions(elements, "assumption", "-d=", variables)
    output_spec += extract_expressions(elements, "negated_bc", "-nbc=", variables)
    if BC != "":
        BC = "_BC" + BC
    # output_filename = f"{PROJECT_PATH}/input-files/" + name + BC + ".spectra"
    output_filename = filename.replace("specifications", "modified-specs")
    output_filename = re.sub(r"/spec$", "/" + name + BC + ".spectra", output_filename)
    output_filename = re.sub(r"\.spec$", ".spectra", output_filename)

    write_file(output_filename, output_spec)
    file_for_cli = pRespondsToS_substitution(output_filename)
    return file_for_cli, output_filename


def extract_expressions(elements, expression_name, expression_type, variables):
    count = 0
    output = ""
    for i, element in enumerate(elements):
        if element == expression_type:
            count += 1
            output += write_formula(count, elements, i, expression_name, variables)
    return output


def spread_formula(gr1_formula):
    string, names, subformulae = replace_brackets_with_names(gr1_formula)
    if len(subformulae) > 1:
        return [contrapositive_antecedent_nexts(gr1_formula)]
    formulae = re.findall(r"([^&-]*->[^&-]*)", subformulae[0])
    outputs = []
    for formula in formulae:
        string_out = re.sub(names[0], formula, string)
        outputs.append(contrapositive_antecedent_nexts(string_out))
    return outputs


def write_formula(expression_count, elements, i, name, variables):
    formula = elements[i + 1]
    formula = strip_qoutes(formula)
    expression_type = name
    if name == "negated_bc":
        expression_type = "assumption"
    gr1_formula = gr_one(formula, variables)
    if len(re.findall("->", gr1_formula)) > 1:
        gr1_list = spread_formula(gr1_formula)
    else:
        gr1_list = [gr1_formula]
    output = ""
    for i, formula_n in enumerate(gr1_list):
        # formula_n = assign_equalities(formula_n, variables)
        suffix = "_" + str(i + 1)
        if len(gr1_list) == 0:
            suffix = ""
        output += expression_type + " -- "
        output += name + str(expression_count) + suffix + "\n\t"
        output += formula_n + ";\n\n"
    return output


def strip_qoutes(text):
    quote_type = text[0]
    if quote_type == text[-1] and quote_type in ['"', "'"]:
        return text[1:-1]
    return text


def extract_variables(elements, keyword, type, output_vars):
    output = ""
    if keyword not in elements:
        return output
    variables = elements[elements.index(keyword) + 1]
    variables = re.search(r"\'([^\']*)\'", variables).group(1)
    variables = variables.split(",")
    for var in variables:
        output_vars.append(var)
        output += type + " boolean " + var + ";\n"
    return output


def read_BCs(folder):
    filename = folder.replace("specifications", "BCs") + "/BCs"
    return read_file_lines(filename)


def translate_case_study(folder, start_time, delete_broken=True, broad=False):
    # folder = f"{PROJECT_PATH}/input-files/case-studies/specifications/arbiter"
    folder = re.sub(r"/$", "", folder)
    filename = folder + "/spec"
    with open("../../output-files/case-studies/case_study_translation.csv", 'a', newline="") as file:
        log = csv.writer(file)
        log_line = [start_time, get_name(filename), ""]
        try:
            spec = read_file_lines(filename)
        except FileNotFoundError:
            log_line.append("Cannot find file")
            log.writerow(log_line)
            return []

        translated_file_cli, learning_file = translate_spec_to_file(spec, filename, BC="")

        if translated_file_cli.find("No_file_written") >= 0:
            log_line.append("pRespondsToS issue")
            log_line.append(re.sub(r"No_file_written:", "", translated_file_cli))
            log.writerow(log_line)
            return []
        spec_is_realizable = realizable(translated_file_cli, True)
        if spec_is_realizable is None:
            log_line.append("Initial Spec Incompatible")
            log.writerow(log_line)
            return []
        if spec_is_realizable:
            log_line.append("Initial Spec Realizable")
            log.writerow(log_line)
            print("Original Spec is realizable")
        else:
            log_line.append("Initial Spec Unrealizable")
            log.writerow(log_line)

        BC_list = read_BCs(folder)

        list_of_specs = check_BC_list(BC_list, filename, spec, start_time, delete_broken, broad)
        if list_of_specs == []:
            print("No GR(1) BC found that creates realizable spec")
            # return []

        return list_of_specs


def split_by_operator(negated_BC, operator="&"):
    string, names, sub_formulae = replace_brackets_with_names(negated_BC)
    conjuncts = string.split(operator)
    return [replace_names_with_formulae(sub_formulae, names, x) for x in conjuncts]


def replace_brackets_with_names(string):
    '''

    :param string:
    :return: string, names, sub_formulae
    '''
    # TODO: what if one sub_formula is contained in another?
    #  Think i solved this with brackets
    contents = list(parenthetic_contents(string))
    sub_formulae = [tup[1] for tup in contents if tup[0] == 0]
    names = ["sf_name_" + str(i) for i in range(len(sub_formulae))]
    for i, sub_formula in enumerate(sub_formulae):
        string = re.sub(re.escape("(" + sub_formula + ")"), "(" + names[i] + ")", string)
    return string, names, sub_formulae


def replace_names_with_formulae(formulae, names, output):
    for i, name in enumerate(names):
        output = re.sub(re.escape(name), formulae[i], output)
    return output


def sort_expression(body):
    string, names, formulae = replace_brackets_with_names(body)
    parts = string.split("|")
    contains_next = [part for part in parts if part.find("X") >= 0]
    no_next = [part for part in parts if part.find("X") < 0]

    head = '|'.join(no_next)
    body = '|'.join(contains_next)
    head = replace_names_with_formulae(formulae, names, head)
    body = replace_names_with_formulae(formulae, names, body)
    return [head, body]
    # for next in contains_next:
    #     no_next.append(next)
    # output = '|'.join(no_next)
    # output = replace_names_with_formulae(formulae, names, output)
    # return ['|'.join(no_next), '|'.join(contains_next)]


def turn_into_rules(BCs):
    # BCs = ['G(F((!m|!p)&X(!h)))']
    prs = re.compile(r"G\(([^F]+)F")
    alwEv = re.compile(r"^GF")
    for i, expression in enumerate(BCs):
        if alwEv.search(expression):
            continue
        if re.search(r"^G", expression) and not re.search(r"F", expression):
            body = re.search(r"^G\((.*)\)$", expression).group(1)
            head_body = sort_expression(body)
            if len(head_body) == 2 and "" not in head_body:
                negated_antecedent = negate_and_simplify(head_body[0])
                converted_expression = "G(" + negated_antecedent + "->" + head_body[1] + ")"
                BCs[i] = converted_expression
                continue
        respond_pattern = prs.search(expression)
        if respond_pattern:
            antecedent = respond_pattern.group(1)
            if antecedent[-1] == "|":
                negated_antecedent = negate_and_simplify(antecedent[0:-1])
                replacement = "G(" + negated_antecedent + "->F"
                converted_expression = prs.sub(replacement, expression)
                BCs[i] = converted_expression
                continue
    return BCs


def add_negated_BC_to_spec(negated_BC, number, filename, spec):
    modified_spec = copy.deepcopy(spec)
    BCs = split_by_operator(negated_BC)
    BCs = turn_into_rules(BCs)
    BCs = ['-nbc="' + bc + '"' for bc in BCs]
    modified_spec.append(''.join(BCs))
    return translate_spec_to_file(modified_spec, filename, BC=str(number))


def gr1_compliant(text, broad=False):
    if broad:
        return True
    return not re.compile(r"M|R|U|W").search(text)


def unravel(formula, variables, formula_mapping, negated_formula_mapping):
    for var in variables:
        pattern = "!" + var
        if re.search(pattern, formula):
            formula = re.sub(pattern, negated_formula_mapping[var], formula)
            formula = simplify_liveness(formula)
        if re.search(var, formula):
            formula = re.sub(var, formula_mapping[var], formula)
            formula = simplify_liveness(formula)
    return formula


def delete_files(list):
    if type(list) == str:
        list = [list]
    for file in list:
        if os.path.exists(file):
            os.remove(file)
        else:
            print(f"Can't find {file} to delete")


def check_BC_list(BC_list, filename, spec, start_time, delete_broken=True, broad=False):
    BC_list = [re.sub(r"\s", "", x) for x in BC_list]
    name = get_name(filename)
    with open("../../output-files/case-studies/case_study_translation.csv", 'a', newline="") as file:
        log = csv.writer(file)

        output_files = []
        for BC in BC_list:
            number = BC_list.index(BC)
            log_line = [start_time, name, str(number)]
            if not gr1_compliant(BC, broad):
                log_line.append("BC Unsupported")
                log_line.append("Contains one of U|W|M|R")
                log.writerow(log_line)
                continue
            negated_BC = negate_and_simplify(BC)
            (temp_file_cli, temp_learning_file) = add_negated_BC_to_spec(negated_BC, number, filename, spec)
            failed = re.search("No_file_written:(.*)", temp_file_cli)
            if failed:
                log_line.append("Incompatible format")
                log_line.append(failed.group(1))
                log.writerow(log_line)
                continue
            realizable_status = realizable(temp_file_cli, True)
            if realizable_status is None:
                log_line.append("Incompatible format")
                log.writerow(log_line)
                if delete_broken:
                    delete_files([temp_file_cli, temp_learning_file])
                continue
            log_line.append("Complete Translation")
            if not realizable_status:
                print("Negated BC produced unrealizable spec:\n" + temp_file_cli)
                log_line.append("Unrealizable")
                log.writerow(log_line)
            else:
                log_line.append("Realizable")
                log.writerow(log_line)
                print("Negated BC produced realizable spec.")
                print("Adding to list of files to process:")
                print(temp_file_cli)
                output_files.append((temp_file_cli, temp_learning_file))
        return output_files


def apply_distributive_ltl_rules(formula):
    # formula = "(level)"
    alw = re.compile(r"^G\(([^\)]*)\)$")
    ev = re.compile(r"^F\(([^\)]*)\)$")
    disjuncts = split_by_operator(formula, "|")
    for i, disjunct in enumerate(disjuncts):
        conjuncts = split_by_operator(disjunct, "&")
        disjuncts[i] = split_by_group_and_rejoin(alw, conjuncts, op="&", temp_op="G")
    return split_by_group_and_rejoin(ev, disjuncts, "|", "F")


def split_by_group_and_rejoin(alw, parts, op, temp_op):
    always = [alw.search(x).group(1) for x in parts if alw.search(x)]
    other = [x for x in parts if not alw.search(x)]
    output = op.join(other) + op + temp_op + "(" + op.join(always) + ")"
    e_op = re.escape(op)
    return re.sub(e_op + temp_op + r"\(\)|^" + e_op + "|" + e_op + "$", "", output)


def negate_and_simplify(original_formula):
    # original_formula = "(F((r&p))|(r&F(s)&F(G(s))))"
    # original_formula = '!G(!g_0|!g_1)'
    # original_formula = '(((!a&(r1|g2)))W((!r2&F(r2)&G(!g2))))'
    original_formula = "(" + strip_unnecessary_brackets(original_formula) + ")"

    contents = list(parenthetic_contents(original_formula))
    if len(contents) <= 1:
        return negate(original_formula)
    levels = max([tup[0] for tup in contents])
    if levels == 0:
        return negate(original_formula)

    formula_mapping, negated_formula_mapping, nest_names = express_formula_as_levels(original_formula)

    key = list(formula_mapping.keys())[0]
    formula = negated_formula_mapping[key]
    variables = list(nest_names.keys())
    output_formula = unravel(formula, variables, formula_mapping, negated_formula_mapping)
    output_formula = simplify_liveness(output_formula)
    return output_formula


def express_formula_as_levels(original_formula):
    '''

    :param contents:
    :param levels:
    :return: formula_mapping, negated_formula_mapping, nest_names
    '''
    nest_names = {}
    formula_mapping = {}
    contents = list(parenthetic_contents(original_formula))
    if len(contents) <= 1:
        return formula_mapping, formula_mapping, nest_names
    levels = max([tup[0] for tup in contents])
    if levels == 0:
        return formula_mapping, formula_mapping, nest_names
    for level in range(levels):
        top_formula = [tup[1] for tup in contents if tup[0] == level]
        nests = [tup[1] for tup in contents if tup[0] == level + 1]
        names = ["level_" + str(level + 1) + "_nest_" + str(i) for i in range(len(nests))]
        # This sorts nests and names in order of length of nest
        nests, names = (list(t) for t in zip(*sorted(zip(nests, names), key=lambda x: len(x[0]), reverse=True)))

        for i, nest in enumerate(nests):
            nest_names[names[i]] = nest

        for i, formula in enumerate(top_formula):
            for j, nest in enumerate(nests):
                formula = re.sub(re.escape("(" + nest + ")"), "(" + names[j] + ")", formula)
            if top_formula[i] in nest_names.values():
                temp_name = list(nest_names.keys())[list(nest_names.values()).index(top_formula[i])]
            else:
                temp_name = top_formula[i]
            if level > 1:
                formula = apply_distributive_ltl_rules(formula)
            formula_mapping[temp_name] = formula
    negated_formula_mapping = {}
    for key in formula_mapping.keys():
        string = formula_mapping.get(key)
        negated_formula_mapping[key] = negate(string)
    for key in nest_names.keys():
        string = nest_names.get(key)
        if key not in negated_formula_mapping.keys():
            negated_formula_mapping[key] = negate(string)
        if key not in formula_mapping.keys():
            formula_mapping[key] = string

    for key in negated_formula_mapping.keys():
        formula_mapping["!" + key] = negated_formula_mapping[key]

    return formula_mapping, negated_formula_mapping, nest_names


def simplify_liveness(output_formula):
    output_formula = re.sub(r"G\(F\(([^\)]*)\)\)", r"GF(\1)", output_formula)
    # contents = list(parenthetic_contents_with_function(output_formula))

    return output_formula
