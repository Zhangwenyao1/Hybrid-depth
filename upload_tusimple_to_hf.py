#!/usr/bin/env python3
"""
Upload Stage1 TuSimple dataset (process_32_order) to Hugging Face: WenyaoZhang/Hybrid-depth

Expected local structure (as in params e.g. basicParams_dino_clip.yaml):
  LOCAL_PROCESS_32_ORDER/
    train_data/
      data_list/
        train.txt
      <images etc.>
    test_data/
      data_list/
        test.txt
      <images etc.>

Usage:
  pip install -U huggingface_hub
  huggingface-cli login
  python upload_tusimple_to_hf.py

Set LOCAL_PROCESS_32_ORDER below (e.g. /data/zhangwenyao/drive_data/TUSimple/process_32_order).
Uploads to the repo under tusimple/train_data and tusimple/test_data.
"""
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "WenyaoZhang/Hybrid-depth"
# Local path to process_32_order (containing train_data/ and test_data/). Change this before running.
LOCAL_PROCESS_32_ORDER = "/path/to/your/TUSimple/process_32_order"
PATH_IN_REPO_TRAIN = "tusimple/train_data"
PATH_IN_REPO_TEST = "tusimple/test_data"


def main():
    root = Path(LOCAL_PROCESS_32_ORDER)
    if not root.is_dir():
        raise FileNotFoundError(
            f"process_32_order not found: {LOCAL_PROCESS_32_ORDER}. "
            "Edit LOCAL_PROCESS_32_ORDER to point to TUSimple/process_32_order "
            "(containing train_data/ and test_data/)."
        )
    train_data = root / "train_data"
    test_data = root / "test_data"
    if not train_data.is_dir():
        raise FileNotFoundError(f"train_data not found under {root}")
    if not test_data.is_dir():
        raise FileNotFoundError(f"test_data not found under {root}")
    if not (train_data / "data_list" / "train.txt").is_file():
        raise FileNotFoundError(f"train_data/data_list/train.txt not found under {root}")
    if not (test_data / "data_list" / "test.txt").is_file():
        raise FileNotFoundError(f"test_data/data_list/test.txt not found under {root}")

    api = HfApi()
    print("Uploading train_data (this may take a while)...")
    api.upload_folder(
        folder_path=str(train_data),
        path_in_repo=PATH_IN_REPO_TRAIN,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print("Uploading test_data...")
    api.upload_folder(
        folder_path=str(test_data),
        path_in_repo=PATH_IN_REPO_TEST,
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Done. Dataset at https://huggingface.co/{REPO_ID}/tree/main/tusimple")


if __name__ == "__main__":
    main()
