# OpenStethoscope — Complete Pipeline Results

**Generated:** 2026-02-23  
**Embeddings:** Google HeAR (512-dim, 2-second windows at 16kHz, no software filters)  
**Classifier:** LogisticRegression (C=1.0, class_weight=balanced) + StandardScaler  
**Evaluation:** StratifiedGroupKFold(n_splits=5), patient as group — OOF AUROC + bootstrap 95% CI

---

## 1. Classifier Performance (Patient-Level)

| Task | OOF AUROC | 95% CI | Fold AUROC | n+ | n− | N |
|------|-----------|--------|------------|----|----|---|
| **COPD vs Healthy** | **0.939** | [0.900–0.970] | 0.945 ± 0.033 | 117 | 61 | 178 |
| **COPD vs All** | **0.890** | [0.848–0.930] | 0.892 ± 0.011 | 117 | 163 | 280 |
| Asthma vs Healthy | 0.877 | [0.798–0.944] | 0.870 ± 0.036 | 34 | 61 | 95 |
| Heart Failure vs Healthy | 0.858 | [0.754–0.945] | 0.868 ± 0.122 | 19 | 61 | 80 |
| Pneumonia vs Healthy | 0.806 | [0.625–0.949] | 0.839 ± 0.178 | 11 | 61 | 72 |



**Primary metric: COPD vs All** (clinically honest — COPD vs mixed sick patients)  
**Secondary metric: COPD vs Healthy** (clean research comparison)

---

## 2. Group Definition

Each group = one unique patient. Patient IDs:
- **ICBHI:** `ICBHI_101` … `ICBHI_226` (3-digit integers from ICBHI_Challenge_diagnosis.txt)
- **Fraiwan:** `FR_P1` … `FR_P112` (from filename prefix BPx/DPx/EPx)
- **RDTR:** `RDTR_H002` … `RDTR_H063` (original dataset IDs)

StratifiedGroupKFold guarantees every recording/window of a patient is in exactly one fold.
No patient appears in both train and test in any fold.

---

## 3. Descriptive Statistics

### 3a. ICBHI 2017 (Portugal + Greece)

| Property | Value |
|----------|-------|
| Patients | 126 |
| Recordings | 920 |
| Total duration | 329.6 min |
| Mean duration/recording | 21.5 ± 8.3 s |
| Min / Max duration | 7.9 s / 86.2 s |
| Recordings per patient | 7.3 ± 8.4 (min 1, max 66) |

**Diagnoses:**
| Label | Patients |
|-------|----------|
| COPD | 64 |
| Healthy | 26 |
| URTI | 14 |
| Bronchiectasis | 7 |
| Bronchiolitis | 6 |
| Pneumonia | 6 |
| LRTI | 2 |
| Asthma | 1 |
| **Total** | **126** |

**Chest locations (unique recordings):**
| Code | Location | Count |
|------|----------|-------|
| Al | Anterior Left | 80 |
| Ar | Anterior Right | 74 |
| Pl | Posterior Left | 61 |
| Pr | Posterior Right | 59 |
| Lr | Lateral Right | 54 |
| Tc | Trachea | 53 |
| Ll | Lateral Left | 42 |

**Devices:**
| Device | Patients |
|--------|----------|
| Meditron Master Elite | 64 |
| AKG C417L (reference mic) | 32 |
| Littmann Classic II SE | 23 |
| Littmann 3200 | 11 |

Note: Some patients have recordings from multiple devices.

---

### 3b. Fraiwan/KAUH (Jordan)

| Property | Value |
|----------|-------|
| Patients | 112 |
| WAV files | 336 (3 per patient: B/D/E Littmann hardware modes) |
| Recordings per patient | 3 (one per filter mode, one chest position) |
| Duration per recording | 5–30 s (per paper; WAVs on Mac, not computed here) |
| Age | 21–90 years (mean 50.5 ± 19.4) |
| Sex | 43 female, 69 male |
| Device | Littmann 3200 (built-in Bluetooth mic) |
| Chest positions | 1 per patient (posterior, various sides/levels) |

**Diagnoses:**
| Label | Patients |
|-------|----------|
| Normal | 35 |
| Asthma | 33 |
| Heart Failure | 19 |
| COPD | 11 |
| Pneumonia | 5 |
| Lung Fibrosis | 4 |
| Bronchitis | 3 |
| Pleural Effusion | 2 |
| **Total** | **112** |

**Filter modes (B/D/E):**  
Each patient has 3 WAVs. Prefixes: B=Bell (20–200 Hz emphasis), D=Diaphragm (100–500 Hz, standard lung), E=Extended (50–500 Hz). These are hardware output modes of the Littmann 3200, not software filters applied by us. All 3 are averaged for one patient-level embedding.

---

### 3c. RDTR — RespiratoryDatabase@TR (Turkey)

| Property | Value |
|----------|-------|
| Patients | 42 (all COPD) |
| Recordings | 504 (12 channels × 42 patients) |
| Channels | L1–L6, R1–R6 (left/right, positions 1–6) |
| Recordings per patient | 12 (all locations) |
| Duration per recording | Not computed (WAVs on Mac) |
| Device | Electronic stethoscope (model not specified in paper) |

**COPD GOLD grades:**
| Grade | Patients |
|-------|----------|
| GOLD 0 (at risk) | 6 |
| GOLD 1 (mild) | 5 |
| GOLD 2 (moderate) | 7 |
| GOLD 3 (severe) | 7 |
| GOLD 4 (very severe) | 17 |
| **Total** | **42** |

Note: All RDTR patients are COPD. No healthy controls. Used as positive class only.

---

### 3d. Composite Dataset

| Property | Value |
|----------|-------|
| **Total patients** | **280** |
| **Total recordings** | **1,760** |
| ICBHI (Portugal/Greece) | 126 pts, 920 recordings |
| Fraiwan (Jordan) | 112 pts, 336 recordings |
| RDTR (Turkey) | 42 pts, 504 recordings |
| Countries | 4 (Portugal, Greece, Jordan, Turkey) |

**Pooled diagnoses across all datasets:**
| Diagnosis | Patients | Source |
|-----------|----------|--------|
| COPD | **117** | ICBHI (64) + Fraiwan (11) + RDTR (42) |
| Normal/Healthy | 61 | ICBHI (26 Healthy) + Fraiwan (35 Normal) |
| Asthma | 34 | ICBHI (1) + Fraiwan (33) |
| URTI | 14 | ICBHI only |
| Heart Failure | 19 | Fraiwan only |
| Bronchiectasis | 7 | ICBHI only |
| Bronchiolitis | 6 | ICBHI only |
| Pneumonia | 11 | ICBHI (6) + Fraiwan (5) |
| LRTI | 2 | ICBHI only |
| Lung Fibrosis | 4 | Fraiwan only |
| Bronchitis | 3 | Fraiwan only |
| Pleural Effusion | 2 | Fraiwan only |
| **Total** | **280** | |

---

## 4. Pipeline Structure

```
pipeline/
├── 01_data/
│   └── compute_descriptive_stats.py   ← generates descriptive_stats.json
├── 02_embeddings/
│   └── embed_with_hear.py             ← embed any dataset with HeAR
├── 03_classifiers/
│   └── train_classifiers.py           ← train + evaluate all classifiers
└── 04_results/
    ├── README.md                       ← this file
    ├── descriptive_stats.json
    └── classifier_performance.json
```

**Raw embeddings used (no software filter applied):**
- `datasets/icbhi_full/icbhi_embeddings_cache.npz` — 920 × 512 (recording-level)
- `datasets/fraiwan_embeddings.npz` — 336 × 512 (WAV-level, 3 per patient)
- `datasets/rdtr_embeddings.npz` — 504 × 512 (channel-level)


