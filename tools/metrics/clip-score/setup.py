import os

import setuptools


def read(rel_path):
    base_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_path, rel_path)) as f:
        return f.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":
    setuptools.setup(
        name="clip-score",
        version=get_version(os.path.join("src", "clip_score", "__init__.py")),
        author="Taited",
        author_email="taited9160@gmail.com",
        description=("Package for calculating CLIP-Score" " using PyTorch"),
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        url="https://github.com/taited/clip-score",
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
        ],
        python_requires=">=3.5",
        entry_points={
            "console_scripts": [
                "clip-score = clip_score.clip_score:main",
            ],
        },
        install_requires=[
            "numpy",
            "pillow",
            "torch>=1.7.1",
            "torchvision>=0.8.2",
            "ftfy",
            "regex",
            "tqdm",
        ],
        extras_require={"dev": ["flake8", "flake8-bugbear", "flake8-isort", "nox"]},
    )
