## Web UI

Start the server:

```bash
python apps/cli/run.py web
```

Then open:
```
http://localhost:5000
```

### Stopping the Server

When you're done, stop the server by pressing `Ctrl+C` in the terminal where it's running.

### User Interface

The web UI guides you through 4 simple steps:

1. **Add Photos** - Upload or drag & drop your face images (minimum 2, recommended 6-10)
2. **Tune Output** - Configure quality, size, and background settings
3. **Process** - Watch the progress as faces are aligned, averaged, and blended
4. **Download** - Get your final averaged face image

### Features

- **Sticky Header** - The step indicator stays visible as you scroll
- **Keyboard Shortcuts** - Press `?` to view available shortcuts
- **Responsive Design** - Works on desktop and mobile devices
- **Progress Tracking** - Real-time progress bar and log output
- **Tooltips & Hints** - Helpful tips below each action button

### Tips

- Use at least 6-10 photos for smoother results
- Use the "Use example photos" button if you want a quick test
- Default preset is Maximum with a Large output size (1600x2200)
- Balanced and Maximum presets require enhancement libraries (installed via `environment.yml`)
- Advanced settings are optional and let you tweak size and sharpening

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Esc` | Close modals / Cancel |
| `?` | Show keyboard shortcuts help |
| `Ctrl+C` | Stop server (in terminal) |

### Privacy

Processing is 100% local - your photos never leave your device. By default the server binds to `127.0.0.1` (localhost). To expose it to your LAN, run:

```bash
python apps/cli/run.py web --host 0.0.0.0
```

Models are downloaded on first use and saved to `models/`. The web UI loads Google Fonts.
