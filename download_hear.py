"""
download_hear.py — download Google HeAR model to local HuggingFace cache.

Requires a free HuggingFace account with the HeAR license accepted:
  https://huggingface.co/google/hear

  export HF_TOKEN=hf_...
  python download_hear.py
"""
import os, pathlib
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
if not token:
    print("Warning: HF_TOKEN not set — download may fail for gated models.")

print("Downloading HeAR...")
path = snapshot_download("google/hear", token=token)
print(f"HeAR cached at: {path}")

# Write path hint for server.py (works in Docker and locally)
for candidate in [pathlib.Path("/app/hear_path.txt"), pathlib.Path(__file__).parent / "hear_path.txt"]:
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(path)
        print(f"Path written to: {candidate}")
        break
    except (PermissionError, OSError):
        continue

print("Done. Run: python app/server.py")
