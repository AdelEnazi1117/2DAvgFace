// Global state
let selectedFiles = [];
let currentJobId = null;
const previewUrls = new Map();

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const fileSummary = document.getElementById('fileSummary');
const fileNotice = document.getElementById('fileNotice');
const uploadBtn = document.getElementById('uploadBtn');
const addMoreBtn = document.getElementById('addMoreBtn');
const clearFilesBtn = document.getElementById('clearFilesBtn');
const cleanCacheBtn = document.getElementById('cleanCacheBtn');
const cleanResetBtn = document.getElementById('cleanResetBtn');
const uploadSection = document.getElementById('uploadSection');
const settingsSection = document.getElementById('settingsSection');
const processingSection = document.getElementById('processingSection');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const processBtn = document.getElementById('processBtn');
const startOverBtn = document.getElementById('startOverBtn');
const errorStartOverBtn = document.getElementById('errorStartOverBtn');
const backToFilesBtn = document.getElementById('backToFilesBtn');
const resetFromSettingsBtn = document.getElementById('resetFromSettingsBtn');
const resetFromProcessingBtn = document.getElementById('resetFromProcessingBtn');
const resultImage = document.getElementById('resultImage');
const downloadBtn = document.getElementById('downloadBtn');
const errorText = document.getElementById('errorText');
const qualityPresetSelect = document.getElementById('qualityPreset');
const sizePresetSelect = document.getElementById('sizePreset');
const useExampleBtn = document.getElementById('useExampleBtn');
const progressText = document.getElementById('progressText');
const progressBar = document.getElementById('progressBar');
const progressPercent = document.getElementById('progressPercent');
const logOutput = document.getElementById('logOutput');
const errorLogOutput = document.getElementById('errorLogOutput');
const cancelBtn = document.getElementById('cancelBtn');
const toggleLogsBtn = document.getElementById('toggleLogsBtn');

// Shortcuts modal elements
const shortcutsModal = document.getElementById('shortcutsModal');
const shortcutsBackdrop = document.getElementById('shortcutsBackdrop');
const closeShortcutsBtn = document.getElementById('closeShortcutsBtn');
const shortcutsBtn = document.getElementById('shortcutsBtn');

const TWO_D_PRESETS = {
    fast: {
        feather: 19,
        sharpen: 0.25,
        exportScale: 1.0
    },
    balanced: {
        feather: 31,
        sharpen: 0.35,
        exportScale: 1.0
    },
    max: {
        feather: 41,
        sharpen: 0.45,
        exportScale: 1.0
    }
};
const SIZE_PRESETS = {
    small: { width: 900, height: 1200 },
    medium: { width: 1200, height: 1600 },
    large: { width: 1600, height: 2200 }
};
let statusPoller = null;
let lastLogs = [];
let usingExamples = false;
let exampleFileCount = 0;
let logsVisible = true;

const ACCEPTED_TYPES = new Set(['image/jpeg', 'image/jpg', 'image/png']);

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    applyQualityPreset();
    applySizePreset();
    renderFileList();
    updateFileSummary();
    setStep('upload');
});

function setupEventListeners() {
    // File upload events
    uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (event) => {
        addFiles(event.target.files);
        fileInput.value = '';
    });

    uploadArea.addEventListener('dragover', (event) => {
        event.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (event) => {
        event.preventDefault();
        uploadArea.classList.remove('drag-over');
        addFiles(event.dataTransfer.files);
    });

    // Button events
    addMoreBtn.addEventListener('click', () => fileInput.click());
    clearFilesBtn.addEventListener('click', clearFiles);
    cleanCacheBtn.addEventListener('click', cleanServerCache);
    cleanResetBtn.addEventListener('click', resetApp);

    fileList.addEventListener('click', (event) => {
        const removeButton = event.target.closest('.file-remove');
        if (!removeButton) return;
        removeFile(removeButton.dataset.key);
    });

    uploadBtn.addEventListener('click', uploadFiles);
    processBtn.addEventListener('click', processImages);

    startOverBtn.addEventListener('click', resetApp);
    errorStartOverBtn.addEventListener('click', resetApp);
    backToFilesBtn.addEventListener('click', returnToUpload);
    resetFromSettingsBtn.addEventListener('click', resetApp);
    resetFromProcessingBtn.addEventListener('click', resetApp);
    if (cancelBtn) {
        cancelBtn.addEventListener('click', cancelJob);
    }

    // Settings events
    if (qualityPresetSelect) {
        qualityPresetSelect.addEventListener('change', applyQualityPreset);
    }
    if (sizePresetSelect) {
        sizePresetSelect.addEventListener('change', applySizePreset);
    }
    if (useExampleBtn) {
        useExampleBtn.addEventListener('click', useExamplePhotos);
    }

    const widthInput = document.getElementById('width');
    const heightInput = document.getElementById('height');
    if (widthInput) widthInput.addEventListener('change', syncSizePreset);
    if (heightInput) heightInput.addEventListener('change', syncSizePreset);

    // Logs toggle
    if (toggleLogsBtn) {
        toggleLogsBtn.addEventListener('click', toggleLogs);
    }

    // Keyboard shortcuts modal
    if (shortcutsBtn) {
        shortcutsBtn.addEventListener('click', showShortcuts);
    }
    if (shortcutsBackdrop) {
        shortcutsBackdrop.addEventListener('click', hideShortcuts);
    }
    if (closeShortcutsBtn) {
        closeShortcutsBtn.addEventListener('click', hideShortcuts);
    }

    // Global keyboard shortcuts
    document.addEventListener('keydown', handleGlobalKeydown);
}

function handleGlobalKeydown(event) {
    // Escape to close modal
    if (event.key === 'Escape') {
        if (!shortcutsModal.classList.contains('hidden')) {
            hideShortcuts();
            event.preventDefault();
        }
    }

    // Question mark to show shortcuts
    if (event.key === '?' && !event.target.matches('input, textarea')) {
        showShortcuts();
        event.preventDefault();
    }
}

function showShortcuts() {
    shortcutsModal.classList.remove('hidden');
}

function hideShortcuts() {
    shortcutsModal.classList.add('hidden');
}

function toggleLogs() {
    logsVisible = !logsVisible;
    if (logOutput) {
        logOutput.style.display = logsVisible ? 'block' : 'none';
    }
}

function setStep(step) {
    document.body.dataset.step = step;
}

function addFiles(files) {
    const incoming = Array.from(files || []);
    let rejected = 0;
    let duplicates = 0;

    if (usingExamples) {
        usingExamples = false;
        exampleFileCount = 0;
    }

    const fileMap = new Map(selectedFiles.map((file) => [fileKey(file), file]));

    incoming.forEach((file) => {
        if (!ACCEPTED_TYPES.has(file.type)) {
            rejected += 1;
            return;
        }
        const key = fileKey(file);
        if (fileMap.has(key)) {
            duplicates += 1;
            return;
        }
        fileMap.set(key, file);
    });

    selectedFiles = Array.from(fileMap.values());
    renderFileList();
    updateFileSummary(rejected, duplicates);
}

function removeFile(key) {
    selectedFiles = selectedFiles.filter((file) => fileKey(file) !== key);
    revokePreview(key);
    renderFileList();
    updateFileSummary();
}

function clearFiles() {
    clearAllPreviews();
    selectedFiles = [];
    renderFileList();
    updateFileSummary();
}

function renderFileList() {
    fileList.innerHTML = '';

    if (selectedFiles.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'file-empty';
        empty.textContent = 'No files selected yet.';
        fileList.appendChild(empty);
        return;
    }

    selectedFiles.forEach((file) => {
        const key = fileKey(file);
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';

        const thumb = document.createElement('div');
        thumb.className = 'file-thumb';

        const preview = ensurePreview(key, file);
        const thumbImg = document.createElement('img');
        thumbImg.src = preview;
        thumbImg.alt = file.name;
        thumb.appendChild(thumbImg);

        const meta = document.createElement('div');
        meta.className = 'file-meta';

        const name = document.createElement('div');
        name.className = 'file-name';
        name.textContent = file.name;

        const size = document.createElement('div');
        size.className = 'file-size';
        const ext = getFileExtension(file);
        if (ext) {
            const tag = document.createElement('span');
            tag.className = 'file-tag';
            tag.textContent = ext.toUpperCase();
            size.appendChild(tag);
        }
        size.append(document.createTextNode(formatFileSize(file.size)));

        meta.appendChild(name);
        meta.appendChild(size);

        const removeBtn = document.createElement('button');
        removeBtn.className = 'file-remove';
        removeBtn.type = 'button';
        removeBtn.dataset.key = key;
        removeBtn.textContent = 'Remove';

        fileItem.appendChild(thumb);
        fileItem.appendChild(meta);
        fileItem.appendChild(removeBtn);
        fileList.appendChild(fileItem);
    });
}

function updateFileSummary(rejected = 0, duplicates = 0) {
    if (usingExamples) {
        fileSummary.textContent = `Example set - ${exampleFileCount} photo${exampleFileCount === 1 ? '' : 's'}`;
        fileNotice.textContent = 'Using example photos from data/examples/faces_example.';
        uploadBtn.disabled = true;
        clearFilesBtn.disabled = true;
        cleanResetBtn.disabled = false;
        return;
    }

    const totalSize = selectedFiles.reduce((sum, file) => sum + file.size, 0);
    fileSummary.textContent = `${selectedFiles.length} file${selectedFiles.length === 1 ? '' : 's'} - ${formatFileSize(totalSize)}`;

    const notices = [];
    if (rejected > 0) {
        notices.push(`${rejected} unsupported file${rejected === 1 ? '' : 's'} skipped`);
    }
    if (duplicates > 0) {
        notices.push(`${duplicates} duplicate${duplicates === 1 ? '' : 's'} ignored`);
    }
    fileNotice.textContent = notices.join(' | ');

    uploadBtn.disabled = selectedFiles.length < 2;
    clearFilesBtn.disabled = selectedFiles.length === 0;
    cleanResetBtn.disabled = selectedFiles.length === 0 && !currentJobId;
}

async function cleanServerCache() {
    const confirmed = window.confirm('This will delete uploaded and generated files on the server. Continue?');
    if (!confirmed) return;

    cleanCacheBtn.disabled = true;
    const originalText = cleanCacheBtn.textContent;
    cleanCacheBtn.textContent = 'Cleaning...';

    try {
        const response = await fetch('/cleanup', { method: 'POST' });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Cleanup failed');
        }
        await response.json();
        resetApp();
        fileNotice.textContent = 'Server cache cleared.';
    } catch (error) {
        showError(`Cleanup failed: ${error.message}`);
    } finally {
        cleanCacheBtn.disabled = false;
        cleanCacheBtn.textContent = originalText;
    }
}

async function useExamplePhotos() {
    if (!useExampleBtn) return;
    useExampleBtn.disabled = true;
    const originalText = useExampleBtn.textContent;
    useExampleBtn.textContent = 'Loading...';

    try {
        const response = await fetch('/examples', { method: 'POST' });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Unable to load examples');
        }
        const data = await response.json();
        currentJobId = data.job_id;
        usingExamples = true;
        exampleFileCount = data.file_count || 0;
        selectedFiles = [];
        clearAllPreviews();
        renderFileList();
        updateFileSummary();
        showSettings();
    } catch (error) {
        showError(`Example set unavailable: ${error.message}`);
    } finally {
        useExampleBtn.disabled = false;
        useExampleBtn.textContent = originalText;
    }
}

function applySizePreset() {
    if (!sizePresetSelect) return;
    const preset = SIZE_PRESETS[sizePresetSelect.value];
    if (!preset) return;
    const width = document.getElementById('width');
    const height = document.getElementById('height');
    if (width) width.value = preset.width;
    if (height) height.value = preset.height;
}

function syncSizePreset() {
    if (!sizePresetSelect) return;
    const width = parseInt(document.getElementById('width')?.value || '0', 10);
    const height = parseInt(document.getElementById('height')?.value || '0', 10);
    const match = Object.entries(SIZE_PRESETS).find(
        ([, size]) => size.width === width && size.height === height
    );
    if (match) {
        sizePresetSelect.value = match[0];
    }
}

function applyQualityPreset() {
    if (!qualityPresetSelect) return;
    const preset = qualityPresetSelect.value;
    const settings = TWO_D_PRESETS[preset];
    if (!settings) return;

    const feather = document.getElementById('feather');
    const sharpen = document.getElementById('sharpen');
    const exportScale = document.getElementById('exportScale');

    if (feather) feather.value = settings.feather;
    if (sharpen) sharpen.value = settings.sharpen;
    if (exportScale) exportScale.value = settings.exportScale;
}

async function uploadFiles() {
    if (selectedFiles.length < 2) {
        showError('Please select at least 2 images.');
        return;
    }

    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<span class="btn-loading"></span> Uploading...';

    try {
        const formData = new FormData();
        selectedFiles.forEach((file) => {
            formData.append('images', file);
        });

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Upload failed');
        }

        const data = await response.json();
        currentJobId = data.job_id;

        uploadBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg> Upload & continue';
        showSettings();
    } catch (error) {
        showError(`Upload failed: ${error.message}`);
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg> Upload & continue';
    }
}

async function processImages() {
    if (!currentJobId) {
        showError('No job ID found. Please upload images first.');
        return;
    }

    const settings = {
        background: document.getElementById('background').value,
        width: parseInt(document.getElementById('width').value, 10),
        height: parseInt(document.getElementById('height').value, 10),
        feather: parseInt(document.getElementById('feather').value, 10),
        sharpen: parseFloat(document.getElementById('sharpen').value),
        exportScale: parseFloat(document.getElementById('exportScale').value),
        qualityPreset: document.getElementById('qualityPreset')?.value || null
    };

    showProcessing();

    try {
        const response = await fetch(`/process/${currentJobId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Processing failed');
        }

        const data = await response.json();

        if (data.status === 'completed') {
            showResult(data.output_url);
        } else if (data.status === 'failed') {
            throw new Error(data.error || 'Processing failed');
        } else {
            startStatusPolling();
        }
    } catch (error) {
        showError(`Processing failed: ${error.message}`);
    }
}

function showSettings() {
    uploadSection.classList.add('hidden');
    settingsSection.classList.remove('hidden');
    processingSection.classList.add('hidden');
    resultSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    setStep('settings');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showProcessing() {
    uploadSection.classList.add('hidden');
    settingsSection.classList.add('hidden');
    processingSection.classList.remove('hidden');
    resultSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    setStep('processing');

    // Reset progress
    if (progressBar) progressBar.style.width = '0%';
    if (progressPercent) progressPercent.textContent = '0%';
    if (progressText) progressText.textContent = 'Initializing face alignment pipeline...';
    if (logOutput) logOutput.textContent = 'Waiting for logs...';
    if (errorLogOutput) errorLogOutput.textContent = 'No logs captured.';

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showResult(imageUrl) {
    processingSection.classList.add('hidden');
    resultSection.classList.remove('hidden');
    setStep('result');
    stopStatusPolling();

    const cacheBuster = `?t=${Date.now()}`;
    resultImage.src = imageUrl + cacheBuster;
    downloadBtn.href = imageUrl + cacheBuster;

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showError(message, logs = null) {
    uploadSection.classList.add('hidden');
    settingsSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    resultSection.classList.add('hidden');
    errorSection.classList.remove('hidden');
    errorText.textContent = message;
    setStep('error');
    stopStatusPolling();

    const finalLogs = logs || lastLogs;
    if (errorLogOutput) {
        if (finalLogs && finalLogs.length) {
            errorLogOutput.textContent = finalLogs.join('\n');
            errorLogOutput.scrollTop = errorLogOutput.scrollHeight;
        } else {
            errorLogOutput.textContent = 'No logs captured.';
        }
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function resetApp() {
    clearAllPreviews();
    selectedFiles = [];
    currentJobId = null;
    usingExamples = false;
    exampleFileCount = 0;
    fileInput.value = '';
    stopStatusPolling();
    lastLogs = [];
    logsVisible = true;

    renderFileList();
    updateFileSummary();

    uploadBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg> Upload & continue';
    uploadBtn.disabled = true;

    uploadSection.classList.remove('hidden');
    settingsSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    resultSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    setStep('upload');

    // Reset progress display
    if (progressBar) progressBar.style.width = '0%';
    if (progressPercent) progressPercent.textContent = '0%';

    document.getElementById('background').value = 'gray';
    if (document.getElementById('qualityPreset')) {
        document.getElementById('qualityPreset').value = 'max';
    }
    if (document.getElementById('sizePreset')) {
        document.getElementById('sizePreset').value = 'large';
    }
    applyQualityPreset();
    applySizePreset();

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function startStatusPolling() {
    stopStatusPolling();
    statusPoller = setInterval(fetchStatus, 1500);
    fetchStatus();
}

function stopStatusPolling() {
    if (statusPoller) {
        clearInterval(statusPoller);
        statusPoller = null;
    }
}

async function fetchStatus() {
    if (!currentJobId) return;
    try {
        const response = await fetch(`/status/${currentJobId}`);
        if (!response.ok) {
            return;
        }
        const data = await response.json();
        updateLogs(data.logs || []);

        // Estimate progress based on logs
        const progress = estimateProgress(data.logs || []);
        if (progressBar) progressBar.style.width = `${progress}%`;
        if (progressPercent) progressPercent.textContent = `${progress}%`;

        if (data.status === 'completed') {
            showResult(data.output_url);
        } else if (data.status === 'failed' || data.status === 'cancelled') {
            showError(data.error || 'Processing failed.', data.logs || []);
        } else {
            if (progressText && data.logs && data.logs.length) {
                progressText.textContent = data.logs[data.logs.length - 1];
            }
        }
    } catch (error) {
        showError(`Status check failed: ${error.message}`);
    }
}

function estimateProgress(logs) {
    if (!logs || logs.length === 0) return 5;

    const logText = logs.join(' ').toLowerCase();

    // Estimate progress based on stage indicators
    if (logText.includes('complete') || logText.includes('export')) {
        return 90;
    } else if (logText.includes('blending') || logText.includes('compositing')) {
        return 70;
    } else if (logText.includes('warping') || logText.includes('warped')) {
        return 50;
    } else if (logText.includes('landmarks') || logText.includes('aligning') || logText.includes('aligned')) {
        return 30;
    } else if (logText.includes('detection') || logText.includes('detected') || logText.includes('reading')) {
        return 15;
    } else if (logText.includes('starting') || logText.includes('initial')) {
        return 5;
    }

    // Fallback: estimate based on log count
    const baseProgress = Math.min(logs.length * 3, 80);
    return Math.max(5, baseProgress);
}

function updateLogs(logs) {
    if (!logOutput) return;
    if (!logs || logs.length === 0) {
        logOutput.textContent = 'Waiting for logs...';
        return;
    }
    lastLogs = logs;
    logOutput.textContent = logs.join('\n');
    logOutput.scrollTop = logOutput.scrollHeight;
}

async function cancelJob() {
    if (!currentJobId) return;
    try {
        const response = await fetch(`/cancel/${currentJobId}`, { method: 'POST' });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'Cancel failed');
        }
        showError('Job cancelled.');
    } catch (error) {
        showError(`Cancel failed: ${error.message}`);
    }
}

function returnToUpload() {
    currentJobId = null;
    usingExamples = false;
    exampleFileCount = 0;
    uploadSection.classList.remove('hidden');
    settingsSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    resultSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    updateFileSummary();
    setStep('upload');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function fileKey(file) {
    return `${file.name}_${file.size}_${file.lastModified}`;
}

function ensurePreview(key, file) {
    if (previewUrls.has(key)) {
        return previewUrls.get(key);
    }
    const url = URL.createObjectURL(file);
    previewUrls.set(key, url);
    return url;
}

function revokePreview(key) {
    const url = previewUrls.get(key);
    if (url) {
        URL.revokeObjectURL(url);
        previewUrls.delete(key);
    }
}

function clearAllPreviews() {
    previewUrls.forEach((url) => URL.revokeObjectURL(url));
    previewUrls.clear();
}

function getFileExtension(file) {
    const parts = file.name.split('.');
    if (parts.length < 2) return '';
    return parts.pop();
}

function formatFileSize(bytes) {
    if (!bytes) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`;
}
