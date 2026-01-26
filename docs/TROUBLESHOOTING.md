## Troubleshooting

### Missing packages

If you see `ModuleNotFoundError`, activate the conda environment and run:

```bash
python scripts/verify_setup.py
```

### No images found

Make sure your folder has at least two jpg/png files:

```bash
python apps/cli/run.py run --faces data/faces/your_set
```

### Enhancement features unavailable

Balanced/Max presets require enhancement packages. Update the environment with:

```bash
conda env update -f environment.yml --prune
```

### Checksum mismatch when downloading models

If you see a checksum mismatch error, the downloaded model does not match the expected hash in `models/manifest.json`.
Delete the model file and retry, or update the manifest if you intentionally want a different version.
