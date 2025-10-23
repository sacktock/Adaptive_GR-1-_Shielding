import re
from spec_repair.util.file_util import read_file_lines, write_file


def violation_to_latex(filename):
    # filename = "../Translators/input-files/examples/lift_violation_modified.txt"
    lines = read_file_lines(filename)
    spec = ''.join(lines)
    timepoints = re.findall(r"holds_at\([^,]*,([^,]*)", spec)
    timepoints = list(dict.fromkeys(timepoints))
    output = "\\begin{flushleft}\n"
    output += "$u = u_" + ' u_'.join(timepoints) + " \in \Sigma$ where $\Sigma = 2^{AP}$:\\\\\n"
    for i in timepoints:
        output += "$u," + str(i) + " \\models "
        # absent = ["!" + x for x in re.findall(r"not_holds_at\(([^,]*)," + str(i), spec)]
        present = re.findall(r"\bholds_at\(([^,]*)," + str(i), spec)
        output += '\\ \\land \\ '.join(present)
        output += "$\\\\\n"
    output += r"\end{flushleft}"
    output_filename = latex_filename(filename)
    write_file(output_filename, output)

    # spec = re.sub(r"not_holds_at\(([^,]*),([^,]*),[^\.]*\.", r"u,\2 \\\\models $!\1$", spec)
    # spec = re.sub(r"holds_at\(([^,]*),([^,]*),[^\.]*\.", r"u,\2 \\\\models $\1$", spec)
    # print(spec)


def spectra_to_latex(filename):
    # filename = "../Translators/input-files/examples/lift.spectra"
    lines = read_file_lines(filename)
    output = "\\begin{flushleft}\n"
    spec = ''.join(lines)
    spec = re.sub(r"boolean ([^;]*);", r"boolean \1;\\\\", spec)
    spec = re.sub(r"env ", "\\\\env{env} ", spec)
    spec = re.sub(r"sys ", "\\\\sys{sys} ", spec)
    spec = re.sub(r" boolean ", " \\\\spec{boolean} ", spec)
    spec = re.sub(r"\nguarantee", "\n\\\\sys{guarantee}$", spec)
    spec = re.sub(r"\nassumption", "\n\\\\env{assumption}$", spec)
    spec = re.sub(r"--([^\n]*)\n", r"--\1$\\\\\n", spec)

    spec = re.sub(r"G\(", "\\\\quad \\\\spec{G}($", spec)
    spec = re.sub(r"GF\(", "\\\\quad \\\\spec{GF}($", spec)
    spec = re.sub(r"F\(", "$\\\\spec{F}($", spec)
    spec = re.sub(r"PREV\(", "$\\\\spec{PREV}($", spec)
    spec = re.sub(r"next\(", "$\\\\spec{next}($", spec)
    spec = re.sub(r";", "$;", spec)

    spec = re.sub(r"->", "\\\\rightarrow ", spec)
    spec = re.sub(r" +", " ", spec)
    spec = re.sub(r"(_| |&)", '\\\\' + r"\1", spec)
    output += spec + r"\end{flushleft}"
    output_filename = latex_filename(filename)
    write_file(output_filename, output)


def asp_to_latex(filename):
    lines = read_file_lines(filename)
    head = True
    for i, line in enumerate(lines):
        if line[0] in ["%", "\n"]:
            lines[i] = ""
            continue
        line = re.sub(r"\n|\t|    ", "", line)
        line = re.sub(r"(_| )", r"\\\1", line)
        is_end = re.search(r"\.", line)
        if head:
            lines[i] = r"\head{" + line + "}\n"
            if not is_end:
                head = False
            continue
        if is_end:
            head = True
            lines[i] = r"    \lastbody{" + line + "}\n\n"
            continue
        lines[i] = r"    \body{" + line + "}\n"
    # print(''.join(lines))
    # output_filename = re.sub(r"([^/]*\.[^/]*)$", r"latex/\1", filename)
    output_filename = latex_filename(filename)
    write_file(output_filename, lines)


def latex_filename(filename):
    output_filename = r"../Translators/latex/" + re.search(r"([^/]*)$", filename).group(1)
    return output_filename
