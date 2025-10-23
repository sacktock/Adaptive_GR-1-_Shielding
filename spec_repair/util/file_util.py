import os
import random
import re
import string
from pathlib import Path
from typing import Optional

# Custom type definitions
Log = str
ASPTrace = str
FilePath = str


def validate_spectra_file(file_path: str) -> None:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != ".spectra":
        raise ValueError(f"Invalid file type: {file_path}. Expected a '.spectra' file.")


def get_line_from_file(filepath: FilePath, line_number: int) -> Optional[str]:
    try:
        with open(filepath, 'r') as f:
            for current_line_num, line in enumerate(f, start=1):
                if current_line_num == line_number:
                    return line.rstrip('\n')
        return None  # If line_number is beyond end of file
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None


def write_file(filename: FilePath, content: list[str]):
    """
    NB: newline = '\\\\n' is necessary so that file is compatible with
    linux (ILASP is run from linux).\n
    :param content: List of lines to save.
    :param filename: filename to save to
    """
    filename = re.sub(r"\\", "/", filename)
    make_directories_if_needed(filename)
    output = ''.join(content)
    with open(filename, "w", newline='\n') as file:
        file.write(output)
        file.close()


def write_to_file(filename: FilePath, content: str):
    make_directories_if_needed(filename)
    with open(filename, 'w') as file:
        file.write(content)


def read_file(file_path: FilePath) -> str:
    with open(file_path, 'r') as file:
        file_content: str = file.read()
    return file_content


def read_file_lines(file_path: FilePath) -> list[str]:
    with open(file_path, "r") as file:
        spec: list[str] = file.readlines()
    return spec


def is_file_format(file_path: str, file_extension: str) -> bool:
    """
    Checks if the path to the (possibly not-existent file) exists,
    then makes sure the extension is the expected one.
    :param file_path: Complete path to a file
    :param file_extension: Expected extension of the file
    :return: True if the file format is expected
    """
    directory, _ = os.path.split(file_path)
    if not os.path.exists(directory):
        return False

    _, extension = os.path.splitext(file_path)
    return extension == file_extension


def generate_filename(spectra_file, replacement, output=False):
    if output:
        spectra_file = spectra_file.replace("input", "output")
    return spectra_file.replace(".spectra", replacement)


def generate_temp_filename(ext):
    assert is_file_extension(ext)
    random_name = generate_random_string(length=10)
    temp_path = os.path.join('/tmp', f"{random_name}{ext}")
    return temp_path


def generate_random_string(length: int = 10) -> str:
    return ''.join(random.choices(string.ascii_letters, k=length))


def is_file_extension(filename: str) -> bool:
    return bool(re.match(r"\.[a-zA-Z0-9_]+$", filename))


def make_directories_if_needed(output_filename):
    folder = re.sub(r"/[^/]*$", "", output_filename)
    if not os.path.isdir(folder):
        # infinite recursion if the path is not a directory, since
        # re.sub(r"/[^/]*$", "", folder) == folder
        make_directories_if_needed(folder)
        os.mkdir(folder)
