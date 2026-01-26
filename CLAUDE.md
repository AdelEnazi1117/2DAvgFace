# CLAUDE.md

## Project Overview

AvgFace is a 2D face averaging tool with a simple CLI and a lightweight web UI.
Core pipeline: `src/average_best.py` (MediaPipe landmarks + face parsing + morphing).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate avgface
python scripts/verify_setup.py
```

## Run

```bash
python apps/cli/run.py run --faces data/faces/your_set
python apps/cli/run.py web
```

Defaults: `--quality max` with 1600x2200 output.

Default outputs are written to `runtime/outputs/`.
