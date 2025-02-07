import re
import os
from tabulate import tabulate


def extract_functions_from_header(header_file):
    functions = []
    with open(header_file, "r") as file:
        content = file.read()

    # Regular expression to match function signatures, including multi-line
    function_pattern = re.compile(r"(\w[\w\s\*&]+)\s+(\w+)\s*\(([^)]*)\)\s*;", re.DOTALL)
    matches = function_pattern.findall(content)
    for match in matches:
        return_type, function_name, params = match
        # Find the line number of the function definition
        for line_num, line in enumerate(content.splitlines(), start=1):
            if function_name in line:
                functions.append((return_type, function_name, params, header_file, line_num))
                break
    return functions


def find_function_implementation(function_name, repo_root):
    for root, _, files in os.walk(repo_root):
        for file in files:
            if file.endswith(".cpp"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, start=1):
                            if re.search(r"\b" + re.escape(function_name) + r"\b\s*\(", line):
                                return file_path, line_num
                except UnicodeDecodeError:
                    continue
    return None, None


def print_functions_table(functions, repo_root):
    table = [
        [
            "ID",
            "Return Type",
            "Function Name",
            "Parameters",
            "Declaration File",
            "Declaration Line",
            "Implementation File",
            "Implementation Line",
        ]
    ]
    for idx, (return_type, function_name, params, decl_file, decl_line) in enumerate(functions, start=1):
        impl_file, impl_line = find_function_implementation(function_name, repo_root)
        table.append([idx, return_type, function_name, params, decl_file, decl_line, impl_file, impl_line])
    print(tabulate(table, headers="firstrow", tablefmt="grid"))


if __name__ == "__main__":
    header_file = "tt_metal/api/tt-metalium/host_api.hpp"  # Updated path to the header file
    repo_root = "/localdev/kmabee/metal3"  # Root directory of your repository
    functions = extract_functions_from_header(header_file)
    print_functions_table(functions, repo_root)
