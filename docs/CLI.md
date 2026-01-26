## CLI Usage

Run the 2D pipeline:

```bash
python apps/cli/run.py run --faces data/faces/your_set
```

Defaults: `--quality max` with a 1600x2200 output size (unless you pass `--quality` or `--size`).
Balanced/Max presets require enhancement libraries (installed via `environment.yml`).

Use the example set:
```bash
python apps/cli/run.py run --examples
```

Quality and size:
```bash
python apps/cli/run.py run --faces data/faces/your_set --quality balanced --size 1200x1600
```

Start the web UI:
```bash
python apps/cli/run.py web
```

### Advanced Options

Any extra flags after `--` are passed directly to `src/average_best.py`:

```bash
python apps/cli/run.py run --faces data/faces/your_set -- --export-scale 1.5 --show
```
