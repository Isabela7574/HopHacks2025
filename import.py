from pathlib import Path
from utils import parse_ravdess_path

# 1. Point to your dataset folder
data_dir = Path("ravdess")  # folder must be in same directory as this file

# 2. Get all files
files = list(data_dir.rglob("*.wav"))  # finds all audio files
print(f"Found {len(files)} audio files!")

# 3. Test on first file
if files:
    example_file = files[0]
    print("Example file:", example_file)
    parsed = parse_ravdess_path(example_file)
    print("Parsed info:", parsed)
