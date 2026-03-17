# OpenStethoscope

**Open-source AI-powered lung auscultation on consumer hardware.**

Build a wireless digital stethoscope for under $100, record lung sounds, and run them through a COPD detection model — locally on your own machine or via the hosted demo.

> ⚠️ **Research demo only.** Not a medical device. Not validated for clinical use. Do not use for clinical decision-making.

---

## What It Does

**Hardware – OpenSteth 1.0:** Build a wireless digital stethoscope using a DJI Mic Mini (approx. $50) and a used stethoscope (approx. $50). Assembly takes about 30 minutes. See [ASSEMBLY.md](ASSEMBLY.md).

**Software — HEAR-COPD:** Uses Google's [HeAR](https://arxiv.org/abs/2403.02522) foundation model (pretrained on 313 million audio clips from YouTube) as a frozen feature extractor. Auscultation sounds are split into 2-second windows, embedded with HeAR, averaged per patient, and scored with a logistic regression classifier. The classifier is trained on standardised (z-scored) 512-dimensional embeddings with L2 regularisation and class-weight balancing to handle dataset imbalance; performance is estimated via 5-fold cross-validation with strict patient-level separation so that no recording from a test patient is ever seen during training.

Pre-trained classifiers are published in this repo.

---

## Performance

Trained on 280 patients across 4 countries (Portugal, Greece, Jordan, Turkey). Patient-level evaluation with StratifiedGroupKFold cross-validation (5 folds).

| Task | AUROC | 95% CI | n+ | n− | n |
|------|-------|--------|----|----|---|
| **COPD vs All patients** *(primary)* | **0.890** | [0.848–0.930] | 117 | 163 | 280 |
| COPD vs Healthy | 0.939 | [0.900–0.970] | 117 | 61 | 178 |
| Asthma vs Healthy | 0.877 | [0.798–0.944] | 34 | 61 | 95 |
| Heart Failure vs Healthy | 0.858 | [0.754–0.945] | 19 | 61 | 80 |
| Pneumonia vs Healthy | 0.806 | [0.625–0.949] | 11 | 61 | 72 |

> **Key insight:** Google's HeAR model was pretrained on a custom corpus of 313M two-second clips extracted from public YouTube videos using a health acoustic event detector (~174k hours) — it has never seen a dedicated lung sound dataset. Its ability to differentiate respiratory pathologies is emergent. A linear probe on top of these frozen embeddings already shows high discriminative accuracy.

---

## Demo

**Live app:** [opensteth.agentic-medicine.com](https://opensteth.agentic-medicine.com)

> Research demo — not a medical device — not validated for clinical use
Important: Noise cancellation must be off for recording with OpenSteth — on the DJI hardware switch, in the DJI Mimo app, and in your phone's microphone settings.
---

## Run Locally

### Prerequisites

- Python 3.9+
- `ffmpeg` installed (`brew install ffmpeg` / `apt install ffmpeg`)
- Free [HuggingFace](https://huggingface.co) account
  - Accept the HeAR model license: [huggingface.co/google/hear](https://huggingface.co/google/hear)
  - Generate an access token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Setup

```bash
git clone https://github.com/petecod/openstethoscope
cd openstethoscope

pip install -r app/requirements.txt

export HF_TOKEN=hf_your_token_here
python download_hear.py        # downloads HeAR model once (~500 MB)

python app/server.py           # starts on http://localhost:8080
```

Open **http://localhost:8080** in your browser. Microphone access works natively on localhost — no HTTPS required.

### Docker

```bash
docker build --build-arg HF_TOKEN=$HF_TOKEN -t openstetho .
docker run -p 8080:8080 openstetho
```

---

## Accessing From Your Phone

Browsers require **HTTPS** for microphone access on any non-localhost address. Three options:

### Option 1 — Tailscale (recommended)
[Tailscale](https://tailscale.com) is a free VPN that gives your machine a trusted HTTPS address. Install it on both your computer and phone, then run:

```bash
tailscale serve --bg http://localhost:8080
```

This gives you a permanent private HTTPS URL (`https://<your-name>.ts.net`) accessible only from your Tailscale devices. No port-forwarding, no public exposure.

### Option 2 — Cloudflare Quick Tunnel (no account needed)
```bash
cloudflared tunnel --url http://localhost:8080
```
Gives a temporary public HTTPS URL. Note: audio passes through Cloudflare's servers.

### Option 3 — ngrok
```bash
ngrok http 8080
```
Similar to Cloudflare tunnel. Free tier available at [ngrok.com](https://ngrok.com).

---

## Privacy

When running locally, all audio is processed on your own machine. Nothing leaves your device.

When using the hosted demo at [opensteth.agentic-medicine.com](https://opensteth.agentic-medicine.com): audio is sent to the inference server (Azure, Sweden Central) solely to compute the AI score, then permanently deleted. No database, no logs, no third-party sharing.

---

## Reproduce the Results

The full training pipeline is in `pipeline/`. You will need to download the original datasets separately (they cannot be redistributed):

| Dataset | Patients | Source |
|---------|----------|--------|
| ICBHI 2017 | 126 | [bhichallenge.med.auth.gr](http://bhichallenge.med.auth.gr) |
| Fraiwan/KAUH | 112 | [Mendeley jwyy9np4gv](https://data.mendeley.com/datasets/jwyy9np4gv) |
| RDTR | 42 | [Mendeley p9z4h98s6j](https://data.mendeley.com/datasets/p9z4h98s6j) |

```bash
# 1. Embed each dataset
python pipeline/02_embeddings/embed_with_hear.py --dataset icbhi  --wav_dir /path/to/icbhi  --out pipeline/02_embeddings/icbhi.npz
python pipeline/02_embeddings/embed_with_hear.py --dataset fraiwan --wav_dir /path/to/fraiwan --out pipeline/02_embeddings/fraiwan.npz
python pipeline/02_embeddings/embed_with_hear.py --dataset rdtr    --wav_dir /path/to/rdtr    --out pipeline/02_embeddings/rdtr.npz

# 2. Train classifiers
python pipeline/03_classifiers/train_classifiers.py

# 3. Results in pipeline/04_results/
```

Full results: [pipeline/04_results/README.md](pipeline/04_results/README.md)

---

## Hardware Assembly

Full guide with photos: [ASSEMBLY.md](ASSEMBLY.md)

**Total cost: <$100 USD, ~30 minutes**

---

## Citation

```bibtex
@software{westarp2026openstethoscope,
  author  = {Westarp, Peter},
  title   = {OpenStethoscope: Open-Source AI Lung Auscultation on Consumer Hardware},
  year    = {2026},
  url     = {https://github.com/petecod/openstethoscope}
}
```

---

## Clinical Validation

Before HEAR-COPD can be considered for routine clinical use, rigorous prospective validation in real-world acute care settings is required to confirm safety, calibration, generalizability, and impact on patient outcomes. The results reported here are from retrospective research datasets and do not substitute for prospective clinical evidence.

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
