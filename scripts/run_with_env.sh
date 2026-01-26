#!/bin/bash
# Helper script that ensures conda environment is activated before running

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

# Get conda base
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Check if environment exists
if ! conda env list | grep -q "^avgface "; then
    echo "Environment 'avgface' not found!"
    echo ""
    echo "Create it with:"
    echo "  conda env create -f environment.yml"
    exit 1
fi

# Activate environment
echo "Activating avgface environment..."
conda activate avgface

# Verify Python version
PYTHON_VERSION=$(python --version)
if [[ ! "$PYTHON_VERSION" =~ "3.11" ]]; then
    echo "Warning: Expected Python 3.11, got: $PYTHON_VERSION"
fi

# Run the script
if command -v avgface >/dev/null 2>&1; then
    echo "Running: avgface $@"
    avgface "$@"
else
    echo "Running: python apps/cli/run.py $@"
    python apps/cli/run.py "$@"
fi
