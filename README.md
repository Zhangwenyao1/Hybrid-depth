<div align="center">

<h3 align="center" style="font-size:24px; font-weight:bold; color:#9C276A; margin: 0;">
  <a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Hybrid-grained_Feature_Aggregation_with_Coarse-to-fine_Language_Guidance_for_Self-supervised_Monocular_ICCV_2025_paper.pdf" style="color:#9C276A; text-decoration: none;">
    Hybrid-grained Feature Aggregation with Coarse-to-fine Language Guidance <br> for Self-supervised Monocular Depth Estimation
  </a>
</h3>

<p align="center">
  ICCV 2025
</p>

<p align="center">
  ⭐ If our project helps you, please give us a star on GitHub to support us!
</p>

<div align="center">
  <a href="https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Hybrid-grained_Feature_Aggregation_with_Coarse-to-fine_Language_Guidance_for_Self-supervised_Monocular_ICCV_2025_paper.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-orange.svg" alt="Paper PDF">
  </a>
  <a href="https://github.com/Zhangwenyao1/Hybrid-depth">
    <img src="https://img.shields.io/badge/Code-GitHub-blue.svg" alt="Code">
  </a>
  <a href="https://huggingface.co/WenyaoZhang/Hybrid-depth/tree/main">
    <img src="https://img.shields.io/badge/🤗-Weights-yellow.svg" alt="Weights">
  </a>
</div>

</div>

If you have any questions about the code, feel free to open an issue.

---

## Clone this repo

```bash
git clone https://github.com/Zhangwenyao1/Hybrid-depth.git
cd Hybrid-depth
```

This repository builds on [Monodepth2](https://github.com/nianticlabs/monodepth2) and [ManyDepth](https://github.com/nianticlabs/manydepth).

---

## Installation

- Python 3.8+
- PyTorch (with CUDA if training)
- Create a conda environment and install dependencies, e.g.:

```bash
conda create -n hybrid_depth python=3.8
conda activate hybrid_depth
pip install torch torchvision
pip install -r requirements.txt  # if present, or install as needed
```

---

## Checkpoints

All pretrained weights can live under the repo root’s **`checkpoints/`** folder. The following are required for training or evaluation.

**DINOv2**  
Download `dinov2_vitb14_pretrain.pth` and place it in `checkpoints/`.
- Hugging Face: <https://huggingface.co/facebook/dinov2-base/tree/main> (download `dinov2_vitb14_pretrain.pth`)

**CLIP**  
The code uses CLIP **RN50**. Place `RN50.pt` in `checkpoints/` (or it will be downloaded there on first run).
- Hugging Face: <https://huggingface.co/jinaai/clip-models/blob/main/RN50.pt>

**Our model weights (Hugging Face: [WenyaoZhang/Hybrid-depth](https://huggingface.co/WenyaoZhang/Hybrid-depth))**

| Content | Path in repo | Usage |
|--------|----------------|-------|
| **Stage2 (paper)** | `checkpoints/` | Download this folder and set `--load_weights_folder` to it for evaluation or single-image inference. |
| **Stage1 pretrained** | `stage1_checkpoint/stage1.ckpt` | Download this file and set `--stage1_checkpoint_path` to it when training Stage2 (if you do not train Stage1 yourself). |


---

## Datasets

- **KITTI**: Prepare the raw dataset as in [Monodepth2](https://github.com/nianticlabs/monodepth2). Set `--data_path` to your KITTI raw root (e.g. `.../kitti_dataset_copy/raw`).

- **TuSimple (Stage1)**: Stage1 uses TuSimple lane data. On [Hugging Face](https://huggingface.co/WenyaoZhang/Hybrid-depth) download **`tusimple/process_32_order.tar.gz`**, extract it, and place the `process_32_order` folder inside a directory named **`TUSimple`** in your data root (e.g. `./data/TUSimple/process_32_order/`). In the Stage1 YAML set **`paths.data_dir`** to that data root and **`tusimple.base_path`** to `TUSimple`. Repo defaults are `./data` and `TUSimple`.

Splits used in this repo: `eigen_zhou`, `eigen_full`, `benchmark`, `odom`, `nyu` (see `Stage2/splits/` and `Stage2/options.py`).

---

## Training

The pipeline has two stages: **Stage1** trains the DINO+CLIP encoder with language–depth alignment (no depth regression); **Stage2** trains the full depth model (Monodepth2-style) using a Stage1 checkpoint. You can either train both stages or use our released Stage1 checkpoint and only train Stage2.

### Stage1 (DINO + CLIP pretraining, PyTorch Lightning)

Stage1 trains the DINO+CLIP encoder with language–depth alignment. It uses PyTorch Lightning and a YAML config; the output is a `.ckpt` file that you pass to Stage2 as `--stage1_checkpoint_path`.

Use **`basicParams_dino_clip_nodepth.yaml`** for Stage1 (TuSimple, alignment only).

```bash
python main.py -c params/basicParams_dino_clip_nodepth.yaml
```

- Set **`paths.data_dir`** to your data root and **`paths.run_dir`** where checkpoints/logs should go. TuSimple paths are under **`tusimple.base_path`** and (optionally) **`tusimple.filenames_file_train`** / **`filenames_file_eval`**.
- Checkpoints are saved under `paths.run_dir` (e.g. `./output`) as `.ckpt` files (`last.ckpt`, `epoch=N.ckpt`). Use one as Stage1 pretrained for Stage2.
- **Local runs:** Copy a config to `*_local.yaml` (e.g. `basicParams_dino_clip_local.yaml`), put your machine paths there, and run with `-c params/basicParams_dino_clip_local.yaml`. Files matching `*_local.yaml` are in `.gitignore` and are not committed.

### Stage2 (Monodepth2-style with DINO/CLIP)

Training with a Stage1 pretrained checkpoint and depth–text alignment (paper setup):

```bash
cd Stage2
python train.py \
  --stage1_checkpoint_path /path/to/stage1_pretrained.ckpt \
  --use_depth_text_align \
  --cat_depth_text_logic \
  --model_name dinoclip_textaligned_dpt_stage1_cat_2 \
  --data_path /path/to/kitti/raw \
  --log_dir /path/to/logs \
  --split eigen_zhou \
  --dataset kitti \
  --height 224 --width 672
```

Set `--stage1_checkpoint_path` to the path of your **Stage1 pretrained** checkpoint (`.ckpt`). If you did not train Stage1, download `stage1_checkpoint/stage1.ckpt` from [Hugging Face](https://huggingface.co/WenyaoZhang/Hybrid-depth/tree/main) and point `--stage1_checkpoint_path` to it. Set `--data_path` and `--log_dir` as needed. See `Stage2/options.py` for more options.



---

## Evaluation

### KITTI depth (Stage2)

To reproduce the paper results, download the **Stage2** checkpoint from [Hugging Face](https://huggingface.co/WenyaoZhang/Hybrid-depth) (the `checkpoints` folder), place it locally, and run the following with `--load_weights_folder` pointing to that folder.

**Recommended arguments (best config from the paper):**

```bash
cd Stage2
python evaluate_depth.py \
  --cat_depth_text_logic \
  --load_weights_folder /path/to/weights \
  --eval_mono \
  --use_depth_text_align \
  --n_depth_text_tokens 256
```

**`--eval_split`** selects which test set to use. Default is **`eigen`** if omitted. Common options:
- **`eigen`** (default) — standard Eigen test set.
- **`eigen_improved`** — alternative test set with improved ground truth.

Add e.g. `--eval_split eigen_improved` to the command above to use a different test set. Full list:

| `--eval_split`    | Test set size | For models trained with...        | Description |
|-------------------|---------------|------------------------------------|-------------|
| `eigen`           | 697           | `--split eigen_zhou` or `eigen_full` | Standard Eigen test files (**default**). |
| `eigen_improved` | 652           | `--split eigen_zhou` or `eigen_full` | Improved ground truth from the [new KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php). |


### Single-image inference and visualization (Stage2)

Run depth prediction on a single image and save a colormapped depth visualization (e.g. `*_disp.jpeg`) and disparity/depth npy:

```bash
cd Stage2
python test_simple.py \
  --image_path /path/to/image.png \
  --model_name /path/to/weights_folder \
  --cat_depth_text_logic \
  --use_depth_text_align \
  --n_depth_text_tokens 256
```

Use the same Stage2 weights folder as in [Evaluation](#kitti-depth-stage2) (e.g. the `checkpoints` folder from [Hugging Face](https://huggingface.co/WenyaoZhang/Hybrid-depth)). Outputs are written to the same directory as the input image by default; set `--vis_dir` to specify an output folder.


---

## Acknowledgement

We thank the authors of [Monodepth2](https://github.com/nianticlabs/monodepth2), [ManyDepth](https://github.com/nianticlabs/manydepth), and [CLIP](https://github.com/openai/CLIP) for their open-source code and models.

---

## Citation

If you use this code in your work, please cite our ICCV 2025 paper:

```bibtex
@inproceedings{zhang2025hybriddepth,
  title     = {Hybrid-grained Feature Aggregation with Coarse-to-fine Language Guidance for Self-supervised Monocular Depth Estimation},
  author    = {Zhang, Wenyao and Liu, Hongsi and Li, Bohan and He, Jiawei and Qi, Zekun and Wang, Yunnan and Zhao, Shengyang and Yu, Xinqiang and Zeng, Wenjun and Jin, Xin},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
