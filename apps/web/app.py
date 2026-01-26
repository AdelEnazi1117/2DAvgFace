#!/usr/bin/env python3
"""
Web GUI for Face Averaging Application
Provides a browser-based interface for uploading images and generating averaged faces.
"""

import sys
import uuid
import shutil
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file

WEB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.config import PROJECT_DESCRIPTION, PROJECT_NAME, PROJECT_TAGLINE, DEFAULT_HOST, DEFAULT_PORT

app = Flask(
    __name__,
    template_folder=str(WEB_ROOT / "templates"),
    static_folder=str(WEB_ROOT / "static"),
)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
app.config['UPLOAD_FOLDER'] = PROJECT_ROOT / "runtime" / "uploads"
app.config['OUTPUT_FOLDER'] = PROJECT_ROOT / "runtime" / "outputs"
app.config['EXAMPLES_FOLDER'] = PROJECT_ROOT / "data" / "examples" / "faces_example"
app.config['ALLOWED_EXTENSIONS'] = {'.jpg', '.jpeg', '.png'}

# Ensure folders exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['OUTPUT_FOLDER'].mkdir(parents=True, exist_ok=True)

# Store job status
jobs = {}
jobs_lock = threading.Lock()

def cleanup_old_files():
    """Remove temporary files older than 1 hour."""
    cutoff = datetime.now() - timedelta(hours=1)

    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
        for item in folder.iterdir():
            try:
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime >= cutoff:
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                print(f"Error cleaning {item}: {e}")


def clear_cache():
    """Remove all files from upload/output folders and reset jobs."""
    removed = {"uploads": 0, "outputs": 0}
    for folder, key in [
        (app.config['UPLOAD_FOLDER'], "uploads"),
        (app.config['OUTPUT_FOLDER'], "outputs"),
    ]:
        for item in folder.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                removed[key] += 1
            except Exception as e:
                print(f"Error cleaning {item}: {e}")
    jobs.clear()
    return removed


def append_log(job_id, message):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        logs = job.setdefault('logs', [])
        logs.append(message.rstrip())
        if len(logs) > 400:
            job['logs'] = logs[-300:]


def finalize_job(job_id, output_path, returncode, stderr=None):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        if job.get('status') == 'cancelled':
            return
        if returncode == 0 and output_path.exists():
            job['status'] = 'completed'
            job['output'] = str(output_path)
            job['output_url'] = f"/result/{job_id}"
        else:
            job['status'] = 'failed'
            job['error'] = stderr or 'Processing failed'


def run_job(job_id, cmd, output_path):
    append_log(job_id, f"Launching: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
    except Exception as exc:
        append_log(job_id, f"Failed to start process: {exc}")
        finalize_job(job_id, output_path, returncode=1, stderr=str(exc))
        return

    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            job['process'] = proc

    try:
        if proc.stdout:
            for line in proc.stdout:
                append_log(job_id, line)
        returncode = proc.wait()
    except Exception as exc:
        append_log(job_id, f"Process error: {exc}")
        returncode = 1

    finalize_job(job_id, output_path, returncode)

def _list_example_images():
    folder = app.config['EXAMPLES_FOLDER']
    if not folder.exists():
        return []
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in app.config['ALLOWED_EXTENSIONS']
    ]
    return sorted(images)


def _create_job_from_paths(paths):
    job_id = str(uuid.uuid4())
    job_folder = app.config['UPLOAD_FOLDER'] / job_id
    job_folder.mkdir(exist_ok=True)

    saved_files = []
    for path in paths:
        filename = path.name.replace('/', '_').replace('\\', '_')
        dest = job_folder / filename
        shutil.copy2(path, dest)
        saved_files.append(str(dest))

    jobs[job_id] = {
        'status': 'uploaded',
        'files': saved_files,
        'created': datetime.now().isoformat()
    }
    return job_id, saved_files


@app.route('/')
def index():
    """Serve the main web interface."""
    cleanup_old_files()
    return render_template(
        'index.html',
        project_name=PROJECT_NAME,
        project_tagline=PROJECT_TAGLINE,
        project_description=PROJECT_DESCRIPTION,
    )

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload and create job."""
    try:
        files = request.files.getlist('images')

        if not files or len(files) == 0:
            return jsonify({'error': 'No files uploaded'}), 400

        # Create unique job ID and folder
        job_id = str(uuid.uuid4())
        job_folder = app.config['UPLOAD_FOLDER'] / job_id
        job_folder.mkdir(exist_ok=True)

        # Save uploaded files
        saved_files = []
        for file in files:
            if file.filename:
                # Secure filename
                filename = file.filename.replace('/', '_').replace('\\', '_')
                if Path(filename).suffix.lower() not in app.config['ALLOWED_EXTENSIONS']:
                    continue
                filepath = job_folder / filename
                file.save(filepath)
                saved_files.append(str(filepath))

        if len(saved_files) < 2:
            shutil.rmtree(job_folder)
            return jsonify({'error': 'Please upload at least 2 images'}), 400

        # Store job info
        jobs[job_id] = {
            'status': 'uploaded',
            'files': saved_files,
            'created': datetime.now().isoformat()
        }

        return jsonify({
            'job_id': job_id,
            'file_count': len(saved_files)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/examples', methods=['POST'])
def load_examples():
    """Create a job from example photos."""
    try:
        images = _list_example_images()
        if len(images) < 2:
            return jsonify({'error': 'Example set is empty. Add images to data/examples/faces_example.'}), 400

        job_id, saved_files = _create_job_from_paths(images)
        return jsonify({'job_id': job_id, 'file_count': len(saved_files)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process/<job_id>', methods=['POST'])
def process(job_id):
    """Process uploaded images to generate averaged face."""
    try:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404

        job = jobs[job_id]

        if job['status'] != 'uploaded':
            return jsonify({'error': 'Job already processing or completed'}), 400

        # Get processing parameters
        data = request.json or {}
        width = data.get('width', 1600)
        height = data.get('height', 2200)
        feather = data.get('feather', 19)
        sharpen = data.get('sharpen', 0.25)
        background = data.get('background', 'gray')
        export_scale = data.get('exportScale', 1.0)
        
        # Enhancement options
        quality_preset = data.get('qualityPreset', None)  # fast, balanced, max

        # Update job status
        job['status'] = 'processing'
        job['pipeline'] = '2d'

        # Prepare output filename
        output_basename = f"averaged_{job_id}"
        output_filename = f"{output_basename}.jpg"
        output_path = app.config['OUTPUT_FOLDER'] / output_filename

        # Build command
        job_folder = app.config['UPLOAD_FOLDER'] / job_id

        cmd = [
            sys.executable, str(PROJECT_ROOT / "src" / "average_best.py"),
            '--faces-dir', str(job_folder),
            '--output-dir', str(app.config['OUTPUT_FOLDER']),
            '--output-name', output_basename,
            '--width', str(width),
            '--height', str(height),
            '--feather', str(feather),
            '--sharpen', str(sharpen),
            '--background', background,
            '--export-scale', str(export_scale),
            '--no-timestamp'
        ]

        if quality_preset in ('fast', 'balanced', 'max'):
            cmd.extend(['--quality', quality_preset])

        # Run processing in background
        with jobs_lock:
            job['status'] = 'processing'
            job['error'] = None
            job['logs'] = []
        append_log(job_id, "Starting 2D pipeline")

        thread = threading.Thread(target=run_job, args=(job_id, cmd, output_path), daemon=True)
        with jobs_lock:
            job['thread'] = thread
        thread.start()

        return jsonify({
            'status': job['status'],
            'output_url': job.get('output_url'),
            'error': job.get('error')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>')
def status(job_id):
    """Get job status."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        payload = {
            'status': job.get('status'),
            'output_url': job.get('output_url'),
            'error': job.get('error'),
            'logs': job.get('logs', [])[-200:],
        }
    return jsonify({
        **payload
    })


@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clear server-side uploads and outputs."""
    try:
        removed = clear_cache()
        return jsonify({
            'status': 'cleared',
            'removed': removed
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cancel/<job_id>', methods=['POST'])
def cancel(job_id):
    """Cancel a running job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    with jobs_lock:
        job = jobs[job_id]
        if job.get('status') not in {'processing', 'uploaded'}:
            return jsonify({'status': job.get('status'), 'error': 'Job not running'}), 400
        job['status'] = 'cancelled'
        job['error'] = 'Cancelled by user'

    proc = job.get('process')
    if proc and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass

    append_log(job_id, "Job cancelled by user.")

    return jsonify({'status': job.get('status')})

@app.route('/result/<job_id>')
def get_result(job_id):
    """Download the result image."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]

    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed'}), 400

    output_path = Path(job['output'])

    if not output_path.exists():
        return jsonify({'error': 'Output file not found'}), 404

    mimetype = 'image/png' if output_path.suffix.lower() == '.png' else 'image/jpeg'
    return send_file(output_path, mimetype=mimetype)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=f"{PROJECT_NAME} web server")
    parser.add_argument('--host', default=DEFAULT_HOST)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print(f"{PROJECT_NAME} Web")
    print("=" * 60)
    print("\nStarting server...")
    print(f"Open your browser and go to: http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    app.run(debug=args.debug, host=args.host, port=args.port)
