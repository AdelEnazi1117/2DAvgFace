# AvgFace

Simple 2D face averaging with a clean CLI and an easy web UI. Processing runs 100% locally on your machine.

## Quick Start

### Web UI

```bash
python apps/cli/run.py web
```

Open http://localhost:5000 and upload photos.

By default the web UI binds to `127.0.0.1` (localhost). To access it from another device on your LAN:

```bash
python apps/cli/run.py web --host 0.0.0.0
```

**Local processing notice:** All image processing happens locally. Models are downloaded on first use and saved in `models/` (hashes pinned in `models/manifest.json`). The web UI loads Google Fonts (you can replace them with local fonts if needed).

**Stop the server:** Press `Ctrl+C` in the terminal.

**Note:** Uses "max" quality mode by default. Enhancement libraries install with `environment.yml`.

### CLI

```bash
python apps/cli/run.py run --examples
```

Result: `runtime/outputs/`

---

## Installation

### Prerequisites

Before you begin, you need:

1. **Conda** (Miniconda or Anaconda)
   - Download from https://docs.conda.io/en/latest/miniconda.html
   - After installation, verify it works:
     ```bash
     conda --version
     ```
   - If you see a version number (e.g., `conda 24.x.x`), you're good!

2. **Git** (optional, but recommended)
   - Download from https://git-scm.com/downloads
   - Verify:
     ```bash
     git --version
     ```

3. **Python 3.11** (will be installed by Conda automatically)

### Step 1: Download the Project

**Option A: Using Git (recommended)**

```bash
git clone https://github.com/your-username/facer-avg.git
cd facer-avg
```

**Option B: Download ZIP**

1. Download the project as ZIP from GitHub
2. Extract the ZIP file
3. Open terminal/command prompt in the extracted folder

### Step 2: Create Conda Environment

This creates an isolated Python environment with all required dependencies.

# Make sure you are in the project folder before running this!

cd facer-avg

```bash
conda env create -f environment.yml
```

What this does:

- Creates a new environment named `avgface`
- Installs Python 3.11
- Installs all required packages (NumPy, OpenCV, MediaPipe, etc.)

### Step 3: Activate the Environment

You must activate the environment every time you want to use AvgFace.

**On Linux/Mac:**

```bash
conda activate avgface
```

**On Windows (Command Prompt):**

```bash
conda activate avgface
```

**On Windows (PowerShell):**

```bash
conda activate avgface
```

You'll see `(avgface)` appear in your terminal prompt, indicating the environment is active.

### Step 4: Verify Installation

Make sure everything is working:

```bash
python scripts/verify_setup.py
```

You should see output like:

```
All dependencies installed.

✓✓✓✓✓ Setup verified successfully! ✓✓✓✓✓
```

If you see errors, check [Troubleshooting](docs/TROUBLESHOOTING.md).

**Models:** Core and enhancement models are downloaded automatically on first use and stored in `models/` (not committed to git).

### After Installation

Every time you want to use AvgFace:

1. Open terminal
2. Activate environment:
   ```bash
   conda activate avgface
   ```
3. Navigate to project folder (if not already there):
   ```bash
   cd facer-avg
   ```
4. Run commands

---

## Usage

### CLI

```bash
python apps/cli/run.py run --faces <path>           # Average faces from folder
python apps/cli/run.py run --examples               # Use bundled example set
python apps/cli/run.py run --faces <path> --quality max --size 1600x2200
```

**Options:**
| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--faces` | path | - | Folder with images (JPG/PNG) |
| `--examples` | - | - | Use `data/examples/faces_example/` |
| `--quality` | `fast`, `balanced`, `max` | `max` | Speed vs quality |
| `--size` | `WIDTHxHEIGHT` | preset | Output resolution |
| `--background` | `gray`, `white`, `black` | `gray` | Background color |

**Quality Modes:**

- `fast`: Quick preview, no enhancement libraries used
- `balanced`: Restoration enabled (requires enhancement libraries)
- `max`: Restoration + upscaling (requires enhancement libraries, default)

### Web UI

```bash
python apps/cli/run.py web
```

Visit http://localhost:5000 and:

- Drop images or select files
- Click "Use example photos" to test
- Download the averaged result

**Stop the server:** Press `Ctrl+C` in the terminal.

#### Web UI Features

- **4-Step Workflow**: Add photos → Tune output → Process → Download
- **Sticky Header**: Step indicator stays visible while scrolling
- **Progress Tracking**: Real-time progress bar and live logs
- **Keyboard Shortcuts**: Press `?` to view available shortcuts
- **Tooltips & Hints**: Helpful guidance below each action
- **Responsive Design**: Works on desktop and mobile

---

## Features

- **Fast & deterministic**: 2D pipeline runs locally
- **Two interfaces**: CLI command line + web UI
- **Private**: No uploads, no third-party servers

---

## Third-Party Notices

See `THIRD_PARTY_NOTICES.md` for dataset, model, and library credits.

## Contributing

See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
- **Quality options**: Fast preview to max quality with upscaling
- **High-quality defaults**: Max quality with restoration + upscaling out of the box

---

## Project Structure

```
facer-avg/
├── apps/
│   ├── cli/run.py          # CLI entrypoint
│   ├── web/app.py          # Web server
│   └── web/templates/      # Web UI (HTML, CSS, JS)
├── src/
│   ├── average_best.py     # Main averaging pipeline
│   └── utils/              # Helper modules
├── data/
│   ├── faces/              # Your input images (gitignored)
│   └── examples/           # Sample faces
├── runtime/outputs/        # Results (gitignored)
└── docs/                   # Detailed docs
```

---

## Documentation

- [Installation Guide](docs/INSTALL.md)
- [CLI Reference](docs/CLI.md)
- [Web UI Guide](docs/WEB.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

---

## License

MIT
