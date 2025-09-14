# train_baseline.py
from pathlib import Path
import random
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from joblib import dump

# -------- Settings --------
DATA_DIR = Path("ravdess")      # dataset folder
SR = 16000
N_MELS = 40
N_PER_CLASS = 50                # speed knob (try 50 or 100; 9999 == all)
RANDOM_SEED = 42
MAX_ERROR_LOGS = 10
# --------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def parse_ravdess_path(path: Path):
    parts = path.stem.split("-")  # MM-VV-EE-II-SS-RR-AA
    if len(parts) != 7:
        raise ValueError(f"Unexpected name format: {path.name}")
    modality, vocal, emotion_id, intensity, statement, repetition, actor = parts
    actor_id = int(actor)
    gender = "male" if actor_id % 2 == 1 else "female"
    return {
        "modality": modality,
        "vocal": vocal,  # '01' speech, '02' song
        "emotion_id": emotion_id,
        "emotion": EMOTION_MAP.get(emotion_id, emotion_id),
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor_id": actor_id,
        "actor_gender": gender,
    }

def featurize(wav_path: Path):
    y, sr = librosa.load(wav_path.as_posix(), sr=SR, mono=True,
                         dtype=np.float32, res_type="kaiser_fast")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    logS = librosa.power_to_db(S, ref=np.max)
    d1 = librosa.feature.delta(logS)
    d2 = librosa.feature.delta(logS, order=2)
    def ms(a): return np.hstack([a.mean(axis=1), a.std(axis=1)])
    return np.hstack([ms(logS), ms(d1), ms(d2)])  # shape (N_MELS*6,)

def select_small_balanced_subset():
    all_wavs = list(DATA_DIR.rglob("*.wav"))
    if not all_wavs:
        raise RuntimeError(f"No .wav files under {DATA_DIR.resolve()}")
    buckets = {name: [] for name in EMOTION_MAP.values()}
    for p in all_wavs:
        try:
            meta = parse_ravdess_path(p)
            if meta["vocal"] != "01":  # speech only
                continue
            emo = meta["emotion"]
            if emo in buckets:
                buckets[emo].append(p)
        except Exception:
            continue
    selected = []
    for emo, files in buckets.items():
        if not files:
            continue
        files = files if len(files) <= N_PER_CLASS else random.sample(files, N_PER_CLASS)
        selected.extend(files)
    random.shuffle(selected)
    if not selected:
        raise RuntimeError("Found WAVs, but none were speech (vocal='01').")
    return selected

def build_table(paths):
    rows, errors = [], []
    print(f"Extracting features from {len(paths)} files...")
    for p in tqdm(paths, desc="Processing", unit="file"):
        try:
            meta = parse_ravdess_path(p)
            x = featurize(p)
            rows.append({"path": p.as_posix(), "emotion": meta["emotion"],
                         **{f"f{i}": v for i, v in enumerate(x)}})
        except Exception as e:
            if len(errors) < MAX_ERROR_LOGS:
                errors.append((p.as_posix(), f"{type(e).__name__}: {e}"))
            continue
    if errors:
        print("\nSkipped files (up to 10):")
        for i, (pp, msg) in enumerate(errors, 1):
            print(f"{i:2d}) {pp}\n    -> {msg}")
    return pd.DataFrame(rows)

def main():
    paths = select_small_balanced_subset()
    print("Selected files:", len(paths))
    df = build_table(paths)

    print(f"\nFeature extraction done! Rows: {len(df)}, Cols: {df.shape[1]}")
    if len(df) == 0:
        print("\n❌ No features were extracted. Ensure files are local (not iCloud).")
        return

    print("\nPer-emotion counts:")
    print(df["emotion"].value_counts().sort_index())

    X = df[[c for c in df.columns if c.startswith("f")]].values
    y = df["emotion"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(max_iter=500, class_weight="balanced")
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("\nReport:\n", classification_report(y_test, y_pred, digits=3))

    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Labels order:", labels.tolist())
    print("Confusion matrix:\n", cm)

    dump({"model": clf, "columns": [c for c in df.columns if c.startswith("f")]},
         "baseline_emotion.joblib")
    df.to_csv("features_sample.csv", index=False)
    print("\nSaved: baseline_emotion.joblib and features_sample.csv")

if __name__ == "__main__":
    main()
