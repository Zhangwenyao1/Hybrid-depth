#!/usr/bin/env python3
"""
Upload Stage1 pretrained checkpoint to Hugging Face: WenyaoZhang/Hybrid-depth

Usage:
  pip install -U huggingface_hub
  huggingface-cli login   # paste your token from https://huggingface.co/settings/tokens
  python upload_stage1_ckpt_to_hf.py

The script uploads the file at LOCAL_CKPT_PATH to the repo as stage1_checkpoint/stage1.ckpt.
Change LOCAL_CKPT_PATH below if your Stage1 .ckpt is elsewhere.
"""
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "WenyaoZhang/Hybrid-depth"
# Local path to your Stage1 pretrained checkpoint (.ckpt). Change this before running.
LOCAL_CKPT_PATH = "/path/to/your/stage1_pretrained.ckpt"
PATH_IN_REPO = "stage1_checkpoint/stage1.ckpt"


def main():
    path = Path(LOCAL_CKPT_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"Stage1 checkpoint not found at {LOCAL_CKPT_PATH}. "
            "Edit LOCAL_CKPT_PATH in this script to point to your .ckpt file."
        )
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=PATH_IN_REPO,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Uploaded {LOCAL_CKPT_PATH} to https://huggingface.co/{REPO_ID}/blob/main/{PATH_IN_REPO}")


if __name__ == "__main__":
    main()
