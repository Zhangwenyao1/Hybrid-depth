#!/usr/bin/env python3
"""
Upload weights_6 to Hugging Face: WenyaoZhang/Hybrid-depth
Usage:
  pip install -U huggingface_hub
  huggingface-cli login   # paste your token from https://huggingface.co/settings/tokens
  python upload_weights_to_hf.py
"""
from huggingface_hub import HfApi

REPO_ID = "WenyaoZhang/Hybrid-depth"
LOCAL_FOLDER = "exps/frozen_textencoder_traindino_len256/models/checkpoints"
PATH_IN_REPO = "checkpoints"  # will appear as weights_6/depth.pth, etc. on HF

def main():
    api = HfApi()
    api.upload_folder(
        folder_path=LOCAL_FOLDER,
        path_in_repo=PATH_IN_REPO,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Uploaded {LOCAL_FOLDER} to https://huggingface.co/{REPO_ID}/tree/main/{PATH_IN_REPO}")

if __name__ == "__main__":
    main()
