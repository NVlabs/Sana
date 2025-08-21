#!/bin/bash
set -e

# download test data
mkdir -p data/data_public
huggingface-cli download Efficient-Large-Model/sana_data_public --repo-type dataset --local-dir ./data/data_public --local-dir-use-symlinks False
huggingface-cli download Efficient-Large-Model/toy_data --repo-type dataset --local-dir ./data/toy_data --local-dir-use-symlinks False
