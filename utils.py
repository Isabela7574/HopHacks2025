# utils.py
import random
from pathlib import Path

# emotion map
EMOTION_MAP = {
    "01": "neutral", 
    "02": "calm", 
    "03": "happy", 
    "04": "sad",
    "05": "angry", 
    "06": "fearful", 
    "07": "disgust", 
    "08": "surprised"
}
def parse_ravdess_path(path: Path):
    parts = path.stem.split("-")  # ['03','01','06','01','02','02','12']
    modality, vocal, emotion_id, intensity, statement, repetition, actor = parts
    actor_id = int(actor)
    gender = "male" if actor_id % 2 == 1 else "female"
    return {
        "modality": modality,      # 03 = audio-only
        "vocal": vocal,            # 01 = speech, 02 = song
        "emotion_id": emotion_id,
        "emotion": EMOTION_MAP[emotion_id],
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
        "actor_id": actor_id,
        "actor_gender": gender
    }
