"""
compute_descriptive_stats.py
Compute descriptive statistics for all 3 datasets used in OpenStethoscope.
Outputs: pipeline/04_results/descriptive_stats.md + descriptive_stats.json
"""

import numpy as np
import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict

ICBHI_WAVS  = Path(os.environ.get("ICBHI_WAVS", "data/icbhi_full/raw/ICBHI_final_database"))
ICBHI_DIAG  = Path(os.environ.get("ICBHI_DIAG", "data/icbhi_full/ICBHI_Challenge_diagnosis.txt"))
ICBHI_NPZ   = Path(os.environ.get("ICBHI_NPZ",  "data/icbhi_full/icbhi_embeddings_cache.npz"))
FRAIWAN_NPZ = Path(os.environ.get("FRAIWAN_NPZ","data/fraiwan_embeddings.npz"))
RDTR_NPZ    = Path(os.environ.get("RDTR_NPZ",   "data/rdtr_embeddings.npz"))

# ─────────────────────────────────────────────────────────────────────────────
# ICBHI
# ─────────────────────────────────────────────────────────────────────────────
print("Computing ICBHI stats...")
import soundfile as sf

diag_map = {}
with open(ICBHI_DIAG) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            diag_map[parts[0]] = parts[1]

# Filename format: {pid}_{cycleN}_{location}_{mode}_{device}.wav
# e.g. 101_1b1_Al_sc_Meditron.wav
# location codes: Al=Anterior Left, Ar=Anterior Right, Ll=Lower Left, etc.
# mode: sc=single channel, mc=multichannel
# device: Meditron, LittC2SE, Litt3200, AKGC417L

icbhi_patients = defaultdict(lambda: {'diag': None, 'recordings': [], 'durations': [],
                                       'locations': set(), 'devices': set(), 'modes': set()})

wav_files = sorted(ICBHI_WAVS.glob("*.wav"))
print(f"  Found {len(wav_files)} WAV files")

for wav in wav_files:
    stem = wav.stem
    parts = stem.split('_')
    if len(parts) < 5:
        continue
    pid = parts[0]
    location = parts[2]  # e.g. Al, Ar, Ll, Lr, Pl, Pr, Tc
    mode = parts[3]      # sc or mc
    device = parts[4]    # Meditron, LittC2SE, Litt3200, AKGC417L

    diag = diag_map.get(pid, 'Unknown')
    icbhi_patients[pid]['diag'] = diag
    icbhi_patients[pid]['locations'].add(location)
    icbhi_patients[pid]['devices'].add(device)
    icbhi_patients[pid]['modes'].add(mode)

    try:
        info = sf.info(str(wav))
        dur = info.duration
        icbhi_patients[pid]['durations'].append(dur)
        icbhi_patients[pid]['recordings'].append(wav.name)
    except Exception as e:
        print(f"  Warning: could not read {wav.name}: {e}")

# Location code → full name
LOC_MAP = {
    'Al': 'Anterior Left', 'Ar': 'Anterior Right',
    'Ll': 'Lateral Left',  'Lr': 'Lateral Right',
    'Pl': 'Posterior Left','Pr': 'Posterior Right',
    'Tc': 'Trachea',
}
DEV_MAP = {
    'Meditron': 'Meditron Master Elite',
    'LittC2SE': 'Littmann Classic II SE',
    'Litt3200': 'Littmann 3200',
    'AKGC417L': 'AKG C417L (reference mic)',
}

all_icbhi_durations = [d for p in icbhi_patients.values() for d in p['durations']]
icbhi_diag_counts   = Counter(p['diag'] for p in icbhi_patients.values())
icbhi_recs_per_pat  = [len(p['recordings']) for p in icbhi_patients.values()]
all_locations       = Counter(loc for p in icbhi_patients.values() for loc in p['locations'])
all_devices         = Counter(dev for p in icbhi_patients.values() for dev in p['devices'])

icbhi_stats = {
    'n_patients': len(icbhi_patients),
    'n_recordings': len(wav_files),
    'diagnoses': dict(icbhi_diag_counts),
    'recordings_per_patient': {
        'mean': float(np.mean(icbhi_recs_per_pat)),
        'std':  float(np.std(icbhi_recs_per_pat)),
        'min':  int(np.min(icbhi_recs_per_pat)),
        'max':  int(np.max(icbhi_recs_per_pat)),
    },
    'duration_seconds': {
        'total': float(np.sum(all_icbhi_durations)),
        'mean':  float(np.mean(all_icbhi_durations)),
        'std':   float(np.std(all_icbhi_durations)),
        'min':   float(np.min(all_icbhi_durations)),
        'max':   float(np.max(all_icbhi_durations)),
    },
    'chest_locations': dict(all_locations),
    'devices': dict(all_devices),
}
print(f"  {icbhi_stats['n_patients']} patients, {icbhi_stats['n_recordings']} recordings")

# ─────────────────────────────────────────────────────────────────────────────
# FRAIWAN
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing Fraiwan stats...")
fr = np.load(FRAIWAN_NPZ, allow_pickle=True)

# Filename: {B/D/E}P{id}_{diagnosis},{sound_type},{position},{age},{sex}.wav
# Position: e.g. "P R M" = Posterior Right Middle
# Parse metadata from filenames
fraiwan_patients = defaultdict(lambda: {'diag': None, 'filters': set(), 'positions': set(),
                                         'age': None, 'sex': None})

for path, pid, diag, ftype in zip(fr['paths'], fr['patient_ids'], fr['diagnoses'], fr['filter_types']):
    pid_str  = str(pid)
    diag_str = str(diag)
    ft_str   = str(ftype)

    fraiwan_patients[pid_str]['diag'] = diag_str
    fraiwan_patients[pid_str]['filters'].add(ft_str)

    # Parse filename for position, age, sex
    fname = Path(str(path)).stem  # e.g. BP1_Asthma,I E W,P R M,70,M
    # Remove leading filter letter + P prefix, then split by _
    m = re.match(r'^[BDE]P\d+_(.*)', fname)
    if m:
        parts = m.group(1).split(',')
        # parts: [diagnosis, sound_type, position, age, sex]
        if len(parts) >= 5:
            fraiwan_patients[pid_str]['positions'].add(parts[2].strip())
            try:
                fraiwan_patients[pid_str]['age'] = int(parts[3].strip())
            except:
                pass
            fraiwan_patients[pid_str]['sex'] = parts[4].strip()

fraiwan_diag_counts = Counter(p['diag'] for p in fraiwan_patients.values())
fraiwan_positions   = Counter(pos for p in fraiwan_patients.values() for pos in p['positions'])
ages = [p['age'] for p in fraiwan_patients.values() if p['age'] is not None]
sexes = Counter(p['sex'] for p in fraiwan_patients.values() if p['sex'])
filter_counts = Counter(str(ft) for ft in fr['filter_types'])

fraiwan_stats = {
    'n_patients': len(fraiwan_patients),
    'n_wav_files': len(fr['paths']),
    'wav_files_per_patient': 3,
    'filter_modes': dict(filter_counts),
    'diagnoses': dict(fraiwan_diag_counts),
    'chest_positions': dict(fraiwan_positions),
    'age': {'mean': float(np.mean(ages)), 'std': float(np.std(ages)),
            'min': int(np.min(ages)), 'max': int(np.max(ages))},
    'sex': dict(sexes),
    'device': 'Littmann 3200 (built-in mic, 3 hardware filter modes: Bell/Diaphragm/Extended)',
    'duration_note': 'WAV files on Mac — durations not computed on Linux. Per paper: 5–30s each.',
    'recording_note': '1 auscultation position per patient; 3 WAVs per patient (B/D/E modes)',
}
print(f"  {fraiwan_stats['n_patients']} patients, {fraiwan_stats['n_wav_files']} WAV files (3 per patient)")

# ─────────────────────────────────────────────────────────────────────────────
# RDTR
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing RDTR stats...")
rd = np.load(RDTR_NPZ, allow_pickle=True)

rdtr_patients = defaultdict(lambda: {'grade': None, 'channels': set()})
for pid, grade, channel in zip(rd['patient_ids'], rd['grades'], rd['channels']):
    pid_str = str(pid)
    rdtr_patients[pid_str]['grade'] = int(grade)
    rdtr_patients[pid_str]['channels'].add(str(channel))

# Count recordings per patient
rdtr_recs_per_patient = Counter(str(pid) for pid in rd['patient_ids'])
recs_per_pat_vals = list(rdtr_recs_per_patient.values())

grade_patient_counts = Counter(p['grade'] for p in rdtr_patients.values())
channel_counts = Counter(str(ch) for ch in rd['channels'])

rdtr_stats = {
    'n_patients': len(rdtr_patients),
    'n_recordings': len(rd['patient_ids']),
    'all_copd': True,
    'gold_grades': {f'GOLD_{k}': v for k,v in sorted(grade_patient_counts.items())},
    'recordings_per_patient': {
        'mean': float(np.mean(recs_per_pat_vals)),
        'std':  float(np.std(recs_per_pat_vals)),
        'min':  int(np.min(recs_per_pat_vals)),
        'max':  int(np.max(recs_per_pat_vals)),
    },
    'channels': dict(sorted(channel_counts.items())),
    'device': 'Electronic stethoscope (model not specified in paper); 10 chest positions per patient',
    'duration_note': 'WAV files on Mac — durations not computed on Linux.',
    'recording_note': '12 channels (L1–L6, R1–R6) per patient, but embeddings have channels listed as L1–L5/R1–R5',
}
print(f"  {rdtr_stats['n_patients']} patients, {rdtr_stats['n_recordings']} recordings")

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing composite stats...")

# For composite: COPD definition and healthy/non-COPD
# ICBHI: COPD=64, Healthy=26, others=36
# Fraiwan: copd=11, normal=35, others=66
# RDTR: all COPD=42, no healthy
composite_copd = icbhi_diag_counts['COPD'] + fraiwan_diag_counts['copd'] + rdtr_stats['n_patients']
composite_healthy = icbhi_diag_counts['Healthy'] + fraiwan_diag_counts['normal']

# All diagnoses across datasets
all_diagnoses = defaultdict(int)
for diag, n in icbhi_diag_counts.items():
    all_diagnoses[diag.lower()] += n
for diag, n in fraiwan_diag_counts.items():
    all_diagnoses[diag.lower()] += n
all_diagnoses['copd'] += rdtr_stats['n_patients']  # RDTR all COPD

composite_stats = {
    'n_patients_total': icbhi_stats['n_patients'] + fraiwan_stats['n_patients'] + rdtr_stats['n_patients'],
    'n_recordings_total': icbhi_stats['n_recordings'] + fraiwan_stats['n_wav_files'] + rdtr_stats['n_recordings'],
    'by_dataset': {'icbhi': icbhi_stats['n_patients'], 'fraiwan': fraiwan_stats['n_patients'], 'rdtr': rdtr_stats['n_patients']},
    'diagnoses_pooled': dict(all_diagnoses),
    'copd_vs_healthy_task': {
        'copd': composite_copd, 'healthy_normal': composite_healthy,
        'note': 'RDTR healthy=0 (excluded from vs_healthy); RDTR copd included as positive'
    },
    'copd_vs_all_task': {
        'copd': composite_copd,
        'non_copd': icbhi_stats['n_patients'] + fraiwan_stats['n_patients'] - icbhi_diag_counts['COPD'] - fraiwan_diag_counts['copd'],
        'total': icbhi_stats['n_patients'] + fraiwan_stats['n_patients'] + rdtr_stats['n_patients'],
    },
    'countries': ['USA/Portugal (ICBHI)', 'Jordan (Fraiwan)', 'Turkey (RDTR)'],
    'devices': list(set(list(DEV_MAP.values()) + [fraiwan_stats['device'], rdtr_stats['device']])),
}

print(f"  Total: {composite_stats['n_patients_total']} patients, {composite_stats['n_recordings_total']} recordings")
print(f"  COPD: {composite_copd}, Healthy/Normal: {composite_healthy}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent.parent / "04_results"
out_dir.mkdir(exist_ok=True)

all_stats = {
    'icbhi':     icbhi_stats,
    'fraiwan':   fraiwan_stats,
    'rdtr':      rdtr_stats,
    'composite': composite_stats,
}
with open(out_dir / "descriptive_stats.json", 'w') as f:
    json.dump(all_stats, f, indent=2, default=str)

print(f"\nSaved to {out_dir / 'descriptive_stats.json'}")
print("\nAll stats computed successfully.")

# Return for use by report generator
if __name__ == '__main__':
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print(f"ICBHI:   {icbhi_stats['n_patients']} pts, {icbhi_stats['n_recordings']} recordings, "
          f"{icbhi_stats['duration_seconds']['total']/60:.1f} min total")
    print(f"Fraiwan: {fraiwan_stats['n_patients']} pts, {fraiwan_stats['n_wav_files']} WAVs (3 modes each)")
    print(f"RDTR:    {rdtr_stats['n_patients']} pts, {rdtr_stats['n_recordings']} recordings")
    print(f"TOTAL:   {composite_stats['n_patients_total']} pts, {composite_stats['n_recordings_total']} recordings")
