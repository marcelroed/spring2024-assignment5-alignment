#!/usr/bin/env bash
set -euo pipefail

echo "Creating virtual environment to run tests"
uv venv ./336_a5_test_venv
source ./336_a5_test_venv/bin/activate
echo "Installing requirements"
uv pip install wheel setuptools pip --upgrade
uv pip install -e .'[test]'
echo "Running tests"
pytest -v ./tests --junitxml=test_results.xml || true
echo "Done running tests"
echo "Cleaning up virtual environment for tests"
deactivate

# Set the name of the output tar.gz file
output_file="cs336-spring2024-assignment-5-submission.zip"
rm "$output_file" || true

# Compress all files in the current directory into a single zip file
zip -r "$output_file" . \
    -x '*egg-info*' \
    -x '*mypy_cache*' \
    -x '*336_a5_test_venv*' \
    -x '*pytest_cache*' \
    -x '*build*' \
    -x '*ipynb_checkpoints*' \
    -x '*__pycache__*' \
    -x '*.pkl' \
    -x '*.pickle' \
    -x '*.txt' \
    -x '*.log' \
    -x '*.json' \
    -x '*.pt' \
    -x '*.safetensors' \
    -x ./data/\* \
    -x '*.out' \
    -x '*.err' \
    -x '.git*' \
    -x '.venv/*' \
    -x 'cutlass/*' \
    -x 'wandb/*' \
    -x '*.bin' \
    -x '*.ckpt'

echo "All files have been compressed into $output_file"
