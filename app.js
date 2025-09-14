const fileInput = document.getElementById('file-input');
const url = URL.createObjectURL(file);
player.src = url;


filePanel.classList.remove('hidden');
actions.classList.remove('hidden');
}


// Drag & drop behavior
['dragenter','dragover'].forEach(evt => {
dropzone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dropzone.classList.add('dragover'); });
});
['dragleave','drop'].forEach(evt => {
dropzone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('dragover'); });
});


dropzone.addEventListener('drop', (e) => {
const dt = e.dataTransfer;
const file = dt && dt.files && dt.files[0];
setFile(file);
});


// Click browse
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => setFile(e.target.files[0]));


// Upload simulation (replace with real API call)
async function fakeUpload(file) {
statusEl.textContent = 'Uploading…';
progress.style.width = '0%';
// Simulate variable speed
let uploaded = 0;
const total = file.size;
return new Promise((resolve) => {
const timer = setInterval(() => {
uploaded += Math.max(total * 0.02, 60_000); // at least ~60KB per tick
const pct = Math.min(100, Math.round((uploaded / total) * 100));
progress.style.width = pct + '%';
if (pct >= 100) {
clearInterval(timer);
resolve();
}
}, 80);
});
}


uploadBtn.addEventListener('click', async () => {
if (!selectedFile) return;
// TODO: replace with real upload
// Example with fetch:
// const form = new FormData();
// form.append('file', selectedFile);
// const res = await fetch('/upload', { method: 'POST', body: form });
// if (!res.ok) { showError('Upload failed'); return; }


uploadBtn.disabled = true;
clearBtn.disabled = true;
await fakeUpload(selectedFile);
uploadBtn.disabled = false;
clearBtn.disabled = false;
statusEl.textContent = 'Upload complete!';
statusEl.classList.add('success');
});


clearBtn.addEventListener('click', resetUI);


// Keyboard focus ring on dropzone via file input focus
fileInput.addEventListener('focus', () => dropzone.style.boxShadow = 'var(--ring)');
fileInput.addEventListener('blur', () => dropzone.style.boxShadow = 'none');