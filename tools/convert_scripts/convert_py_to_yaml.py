import os

import yaml


def convert_py_to_yaml(py_file_path):
    with open(py_file_path, encoding="utf-8") as py_file:
        py_content = py_file.read()

    local_vars = {}
    exec(py_content, {}, local_vars)

    yaml_file_path = os.path.splitext(py_file_path)[0] + ".yaml"

    with open(yaml_file_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(local_vars, yaml_file, default_flow_style=False, allow_unicode=True)


def process_directory(path):
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(".py"):
                py_file_path = os.path.join(root, filename)
                convert_py_to_yaml(py_file_path)
                print(f"convert {py_file_path} to YAML format")


if __name__ == "__main__":
    process_directory("../configs/")
