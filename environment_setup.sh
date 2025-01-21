#!/usr/bin/env bash
set -e

# Default virtual environment directory
VENV_DIR=".venv"

# Check if the virtual environment needs to be created
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR' using uv..."
    uv venv $VENV_DIR
else
    echo "Virtual environment '$VENV_DIR' already exists. Skipping creation."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Update pip to the latest version
echo "Updating pip to the latest version..."
uv pip install -U pip

# Install required dependencies
echo "Installing dependencies..."
uv pip install -r pyproject.toml

# install torchprofile
# uv pip install git+https://github.com/zhijian-liu/torchprofile

echo "Environment setup completed."
