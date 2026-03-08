#!/usr/bin/env python3
"""Upload process_32_order.tar.gz to Hugging Face: WenyaoZhang/Hybrid-depth (tusimple/process_32_order.tar.gz)."""
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "WenyaoZhang/Hybrid-depth"
LOCAL_TAR = Path(__file__).resolve().parent / "process_32_order.tar.gz"
PATH_IN_REPO = "tusimple/process_32_order.tar.gz"


def main():
    if not LOCAL_TAR.is_file():
        raise FileNotFoundError(f"Not found: {LOCAL_TAR}. Run compression first.")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(LOCAL_TAR),
        path_in_repo=PATH_IN_REPO,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Uploaded to https://huggingface.co/{REPO_ID}/blob/main/{PATH_IN_REPO}")


if __name__ == "__main__":
    main()
