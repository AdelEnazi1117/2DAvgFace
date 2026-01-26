## Install (Conda)

```bash
conda env create -f environment.yml
conda activate avgface
python scripts/verify_setup.py
```

`environment.yml` includes the enhancement libraries required for the default max-quality preset.
Models are downloaded on first use and tracked in `models/manifest.json`.

All done! You can now run AvgFace directly:

```bash
python apps/cli/run.py run --faces data/faces/your_set
python apps/cli/run.py web
```
