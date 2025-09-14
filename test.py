from pathlib import Path
import librosa

files = list(Path("ravdess").rglob("*.wav"))
print("wav count:", len(files))
if not files:
    raise SystemExit("No .wav files found! Check your dataset folder.")

p = files[0]
print("sample:", p)

try:
    y, sr = librosa.load(p.as_posix(), sr=16000, mono=True)
    print(f"✅ Loaded OK: {len(y)} samples at {sr} Hz")
except Exception as e:
    print("❌ ERROR loading file:", type(e).__name__, e)
