# Contributing

Thanks for your interest in improving AvgFace!

## Quick start

```bash
conda env create -f environment.yml
conda activate avgface
python scripts/verify_setup.py
```

Run the CLI:

```bash
python apps/cli/run.py run --examples
```

Run the web UI:

```bash
python apps/cli/run.py web
```

## Development checks

```bash
scripts/run_project_checks.sh
```

## Pull requests

- Keep changes focused and avoid unrelated refactors.
- Update docs if behavior or flags change.
- If you add new models or weights, update `models/manifest.json` and `THIRD_PARTY_NOTICES.md`.
- Include a short test/verification note in your PR description.
