import os
import json
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = os.environ.get("DATA_DIR", "./data")
target_data_dir = os.path.join(DATA_DIR, "medical_asr_converted")
os.makedirs(target_data_dir, exist_ok=True)

wavs_dir = os.path.join(target_data_dir, "wavs")
os.makedirs(wavs_dir, exist_ok=True)

def convert_to_manifest(dataset_split, manifest_path, split_name):
    """Convert dataset split to NeMo-compatible manifest with local .wav files."""
    with open(manifest_path, 'w') as fout:
        for i, sample in enumerate(tqdm(dataset_split, desc=f"Creating manifest at {manifest_path}")):
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            transcript = sample["transcription"]

            # Save audio to local .wav file
            local_wav_path = os.path.join(wavs_dir, f"{split_name}_{i}.wav")
            sf.write(local_wav_path, audio_array, sampling_rate)

            duration = len(audio_array) / sampling_rate

            metadata = {
                "audio_filepath": local_wav_path,
                "duration": round(duration, 3),
                "text": transcript
            }

            json.dump(metadata, fout)
            fout.write("\n")

# Load dataset
dataset = load_dataset("jarvisx17/Medical-ASR-EN")

# Split
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_val_split = split_dataset["train"].train_test_split(test_size=0.1111, seed=42)

train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]
test_dataset = split_dataset["test"]

# Paths
train_manifest_path = os.path.join(target_data_dir, "train_manifest.json")
val_manifest_path = os.path.join(target_data_dir, "val_manifest.json")
test_manifest_path = os.path.join(target_data_dir, "test_manifest.json")

# Create manifests with local .wav saving
convert_to_manifest(train_dataset, train_manifest_path, "train")
convert_to_manifest(val_dataset, val_manifest_path, "val")
convert_to_manifest(test_dataset, test_manifest_path, "test")

print(f"✅ Audio and manifests saved in: {target_data_dir}")
