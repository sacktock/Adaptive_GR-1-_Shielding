import re

from spec_repair.config import FASTLAS, MAX_ASP_HYPOTHESES, PROJECT_PATH
from spec_repair.enums import ExpType
from spec_repair.old.specification_helper import run_subprocess, create_cmd
from spec_repair.util.spec_util import run_clingo_raw
from spec_repair.util.file_util import generate_filename, generate_temp_filename, write_to_file, read_file_lines, \
    write_file


def integrate_pylasp(las_file):
    pylasp_script = generate_pylasp_script(las_file)
    pylasp_script = edit_pylasp_for_many_solutions(pylasp_script, MAX_ASP_HYPOTHESES)
    append_pylasp_script(las_file, pylasp_script)


def generate_pylasp_script(las_file):
    cmd = create_cmd(['ILASP', '--version=4', las_file, '-p'])
    pylasp_script = run_subprocess(cmd)
    if pylasp_script == "b''":
        raise ValueError("ILASP Error! No pylasp_script returned!")
    return pylasp_script


def edit_pylasp_for_many_solutions(output, n_solutions):
    output = re.sub(r"^b\"", "", output)
    output = re.sub("\"$", "", output)
    max_sol = r"\nmax_solutions = " + str(n_solutions) + "\n\n\\1"
    output = re.sub(r"(ilasp\.cdilp\.initialise\(\))", max_sol, output)
    output = re.sub(r"while c_egs and solve_result is not None:",
                    r"solution_count = 0\n\nwhile solution_count < max_solutions and solve_result is not None:\n  if c_egs:",
                    output)
    lines = output.split("\n")
    shift = False
    for i, line in enumerate(lines):
        if line == "  ce = ilasp.get_example(c_egs[0]['id'])":
            shift = True
        if shift:
            lines[i] = "  " + line
        if line == "    ilasp.cdilp.add_coverage_constraint(constraint, [ce['id']])":
            shift = False
    output = '\n'.join(lines)
    input_lines = read_file_lines(f"{PROJECT_PATH}/files/input_text_for_pylasp.txt")
    output = re.sub(r"(    c_egs = ilasp\.find_all_counterexamples\(solve_result\))",
                    r"\1\n" + ''.join(input_lines),
                    output)
    output = re.sub(r"if solve_result:\n  print\(ilasp.hypothesis_to_string\(solve_result\['hypothesis'\]\)\)\nelse:",
                    r"if solution_count == 0:",
                    output)
    return output


def append_pylasp_script(las_file, pylasp_script):
    program = read_file_lines(las_file)
    program = [pylasp_script] + program
    write_file(las_file, program)


def error_check_ILASP_output(output):
    asp_tool_name = 'FASTLAS' if FASTLAS else 'ILASP'
    if output == "Timeout":
        raise TimeoutError(f"{asp_tool_name} timed out during run!")
    if output == "b''":
        raise ValueError(f"{asp_tool_name} Error! No output returned!")
    if "error" in output.lower():
        raise ModuleNotFoundError(output)


# TODO: Make it throw the error it returns on bad returns (i.e. syntax errors)
# TODO: check if spectra_file provided should be original version or fixed version
# TODO: don't rename inside of function, provide exact file names and assert their existence
def run_clingo(asp: str) -> list[str]:
    asp_file = generate_temp_filename(ext=".lp")
    write_to_file(asp_file, asp)
    output = run_clingo_raw(asp_file)
    output = output.split("\n")
    for i, line in enumerate(output):
        if len(line) > 100:
            output[i] = '\n'.join(line.split(" "))
    return output


def run_ILASP_raw(las_file, pylasp_integrated=False):
    """
    Runs ILASP on output from encode_ILASP.

    :param las_file: Path to original Spectra specification.
    :param pylasp_integrated: TODO
    :return: Path to file containing learned hypothesis.
    """

    if not pylasp_integrated and not FASTLAS:
        integrate_pylasp(las_file)

    if FASTLAS:
        cmd = create_cmd(["FastLAS", "--nopl", "--force-safety", las_file])
    else:
        cmd = create_cmd(['ILASP', las_file])
    output = run_subprocess(cmd, timeout=60)
    error_check_ILASP_output(output)
    return output


def run_ILASP(las, pylasp_integrated=False):
    """
    Runs ILASP on output from encode_ILASP.

    :param las: Path to original Spectra specification.
    :param pylasp_integrated: TODO
    :return: Path to file containing learned hypothesis.
    """
    ilasp_file = generate_temp_filename(ext=".las")
    write_to_file(ilasp_file, las)
    output = run_ILASP_raw(ilasp_file, pylasp_integrated)
    return output


def get_violations(asp, exp_type: ExpType) -> list[str]:
    output = run_clingo(asp)
    return list(filter(re.compile(rf"violation_holds\(|{str(exp_type)}\(|entailed\(").search, output))