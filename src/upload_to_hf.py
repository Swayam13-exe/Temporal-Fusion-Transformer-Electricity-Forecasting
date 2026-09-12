"""
Run this ONCE to upload the trained model artifacts to Hugging Face Hub,
so cloud deployments (Streamlit Community Cloud) can download them at
startup instead of requiring them to be committed to git.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   # paste a token with WRITE access, from
                             # https://huggingface.co/settings/tokens

Usage:
    python -m src.upload_to_hf --repo_id yourusername/tft-electricity-forecasting
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main(repo_id: str):
    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True)

    ckpts = list(Path("models").glob("tft_best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in models/ -- train the model first.")
    checkpoint_path = max(ckpts, key=lambda p: p.stat().st_mtime)

    schema_path = Path("models/training_dataset.pkl")
    if not schema_path.exists():
        raise FileNotFoundError(
            "models/training_dataset.pkl not found -- run "
            "`python -m src.export_serving_artifacts` first."
        )

    print(f"Uploading {checkpoint_path.name} to {repo_id} ...")
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="tft_best.ckpt",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploading {schema_path.name} to {repo_id} ...")
    api.upload_file(
        path_or_fileobj=str(schema_path),
        path_in_repo="training_dataset.pkl",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Done. Model repo: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True,
                         help="e.g. yourusername/tft-electricity-forecasting")
    args = parser.parse_args()
    main(args.repo_id)