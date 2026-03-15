"""
embed_with_hear.py

Embed raw audio from ICBHI / Fraiwan / RDTR using Google HeAR.
Outputs 512-dim patient-level embeddings (mean over 2-second windows at 16 kHz).

Usage:
  python embed_with_hear.py --dataset icbhi   --wav_dir /path/to/icbhi  --out icbhi.npz
  python embed_with_hear.py --dataset fraiwan --wav_dir /path/to/fraiwan --out fraiwan.npz
  python embed_with_hear.py --dataset rdtr    --wav_dir /path/to/rdtr    --out rdtr.npz

Datasets (download separately — see README):
  ICBHI 2017:   bhichallenge.med.auth.gr
  Fraiwan/KAUH: data.mendeley.com/datasets/jwyy9np4gv
  RDTR:         data.mendeley.com/datasets/p9z4h98s6j
"""

import argparse
import json
import numpy as np
import os
import re
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from scipy.signal import resample as scipy_resample

SR  = 16000   # HeAR expected sample rate
WIN = 32000   # 2-second window at 16kHz


def load_hear_model(model_path: str = None):
    import tensorflow as tf
    if model_path is None:
        # Auto-detect on Linux
        env_path = os.environ.get("HEAR_PATH")
        if env_path:
            return tf.saved_model.load(env_path).signatures
        # Auto-detect HuggingFace cache
        base = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--hear" / "snapshots"
        candidates = sorted(base.glob("*/"))
        if not candidates:
            raise FileNotFoundError("HeAR model not found. Set --model_path.")
        model_path = str(candidates[-1])
    model = tf.saved_model.load(model_path)
    return model


def audio_to_windows(wav_path: str) -> np.ndarray:
    """Load WAV, resample to 16kHz, split into 2s windows. Returns (N, 32000) array."""
    try:
        y, sr = sf.read(wav_path, dtype='float32')
    except Exception as e:
        print(f"  Warning: could not read {wav_path}: {e}")
        return np.zeros((1, WIN), dtype=np.float32)

    if y.ndim > 1:
        y = y.mean(axis=1)  # mono

    if sr != SR:
        y = scipy_resample(y, int(len(y) * SR / sr)).astype(np.float32)

    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))

    windows = []
    for start in range(0, len(y) - WIN + 1, SR):
        windows.append(y[start:start + WIN])
    if not windows:
        windows.append(y[:WIN])

    return np.array(windows, dtype=np.float32)


def embed_windows(model, windows: np.ndarray) -> np.ndarray:
    """Run HeAR on (N, 32000) windows, return (N, 512) embeddings."""
    import tensorflow as tf
    embeddings = model.signatures['serving_default'](tf.constant(windows))
    key = list(embeddings.keys())[0]
    return embeddings[key].numpy()


def embed_dataset(dataset: str, wav_dir: str, model, diag_file: str = None):
    """
    Returns dict: {patient_id: {'embedding': np.array(512,), 'label': str, 'metadata': dict}}
    Patient embedding = mean over all windows from all recordings.
    """
    wav_dir = Path(wav_dir)
    patients = defaultdict(lambda: {'windows': [], 'label': None, 'metadata': {}})

    if dataset == 'icbhi':
        diag_map = {}
        with open(diag_file) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    diag_map[parts[0]] = parts[1]

        for wav in sorted(wav_dir.glob("*.wav")):
            stem = wav.stem.split('_')
            if len(stem) < 5:
                continue
            pid      = stem[0]
            location = stem[2]
            mode     = stem[3]
            device   = stem[4]
            diag     = diag_map.get(pid, 'Unknown')

            windows = audio_to_windows(str(wav))
            patients[pid]['windows'].append(windows)
            patients[pid]['label'] = diag
            patients[pid]['metadata'].setdefault('locations', set()).add(location)
            patients[pid]['metadata'].setdefault('devices', set()).add(device)

    elif dataset == 'fraiwan':
        # Fraiwan: 3 WAVs per patient (B/D/E modes). Use ALL 3 for embedding.
        # Filename: {B/D/E}P{id}_{diagnosis},{sound},{position},{age},{sex}.wav
        for wav in sorted(wav_dir.glob("*.wav")):
            m = re.match(r'^([BDE])P(\d+)_(.*)', wav.stem)
            if not m:
                continue
            ftype = m.group(1)
            pid   = f"P{m.group(2)}"
            rest  = m.group(3).split(',')
            diag  = rest[0].strip().lower().replace(' ', '_') if rest else 'unknown'

            windows = audio_to_windows(str(wav))
            patients[pid]['windows'].append(windows)
            patients[pid]['label'] = diag
            patients[pid]['metadata'].setdefault('filter_types', set()).add(ftype)
            if len(rest) >= 3:
                patients[pid]['metadata']['position'] = rest[2].strip()

    elif dataset == 'rdtr':
        # RDTR: multiple channels (L1-L6, R1-R6) per patient. Filename: {pid}_{channel}.wav
        # Labels (GOLD grades) must be provided via a separate label file.
        label_file = Path(wav_dir) / "labels.json"
        if not label_file.exists():
            raise FileNotFoundError(f"RDTR requires {label_file} with {{patient_id: grade}} mapping.")
        labels = json.load(open(label_file))

        for wav in sorted(wav_dir.glob("*.wav")):
            parts = wav.stem.split('_')
            if len(parts) < 2:
                continue
            pid     = parts[0]
            channel = parts[1] if len(parts) > 1 else 'unknown'
            grade   = labels.get(pid, labels.get(wav.stem, 'unknown'))

            windows = audio_to_windows(str(wav))
            patients[pid]['windows'].append(windows)
            patients[pid]['label'] = f"COPD_GOLD{grade}"
            patients[pid]['metadata'].setdefault('channels', set()).add(channel)

    # Embed each patient
    print(f"  Embedding {len(patients)} patients...")
    results = {}
    for i, (pid, data) in enumerate(sorted(patients.items())):
        all_windows = np.vstack(data['windows'])  # (total_windows, WIN)
        embs = embed_windows(model, all_windows)  # (total_windows, 512)
        patient_emb = embs.mean(axis=0)           # (512,) — patient-level mean
        results[pid] = {
            'embedding': patient_emb,
            'label':     data['label'],
            'n_windows': len(all_windows),
            'metadata':  {k: list(v) if isinstance(v, set) else v
                          for k, v in data['metadata'].items()}
        }
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(patients)} done")

    return results


def save_npz(results: dict, out_path: str):
    """Save patient-level embeddings to npz."""
    patient_ids = np.array(list(results.keys()))
    embeddings  = np.array([results[pid]['embedding'] for pid in patient_ids])
    labels      = np.array([results[pid]['label']     for pid in patient_ids])
    n_windows   = np.array([results[pid]['n_windows'] for pid in patient_ids])
    metadata    = np.array([json.dumps(results[pid]['metadata']) for pid in patient_ids])

    np.savez(out_path,
             patient_ids=patient_ids,
             embeddings=embeddings,
             labels=labels,
             n_windows=n_windows,
             metadata=metadata)
    print(f"  Saved {len(patient_ids)} patient embeddings → {out_path}")
    print(f"  Shape: {embeddings.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    required=True, choices=['icbhi','fraiwan','rdtr'])
    parser.add_argument('--wav_dir',    required=True)
    parser.add_argument('--out',        required=True)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--diag_file',  default=None, help='ICBHI diagnosis .txt file')
    args = parser.parse_args()

    print(f"Loading HeAR model...")
    model = load_hear_model(args.model_path)

    print(f"Embedding {args.dataset}...")
    results = embed_dataset(args.dataset, args.wav_dir, model, args.diag_file)

    save_npz(results, args.out)
    print("Done.")
