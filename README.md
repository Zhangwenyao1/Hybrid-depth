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

## Datasets

- **KITTI**: Prepare the raw dataset as in [Monodepth2](https://github.com/nianticlabs/monodepth2). Set `--data_path` to your KITTI raw root (e.g. `.../kitti_dataset_copy/raw`).


Splits used in this repo: `eigen_zhou`, `eigen_full`, `benchmark`, `odom`, `nyu` (see `Stage2/splits/` and `Stage2/options.py`).

---

## Training

### Stage2 (Monodepth2-style with DINO/CLIP)

```bash
cd Stage2
python train.py \
  --data_path /path/to/kitti/raw \
  --log_dir /path/to/logs \
  --model_name your_exp_name \
  --split eigen_zhou \
  --dataset kitti \
  --height 224 --width 672
```

Main options (see `Stage2/options.py`): `--depth_model_type`, `--pose_model_type`, `--only_dino` / `--only_clip`, `--use_depth_text_align`, etc.

### ManyDepth (multi-frame)

```bash
cd manydepth/manydepth
python train.py \
  --data_path /path/to/kitti/raw \
  --log_dir /path/to/logs \
  --model_name your_exp_name
```

---

## Evaluation

### KITTI depth (Stage2)

**Recommended setup (best config from the paper)**: use the following arguments with your downloaded or trained weights:

```bash
cd Stage2
python evaluate_depth.py \
  --cat_depth_text_logic \
  --load_weights_folder /path/to/weights \
  --eval_mono \
  --use_depth_text_align \
  --n_depth_text_tokens 256
```

**Checkpoint for paper results**: Weights that reproduce the results in the paper are available at [Hugging Face](https://huggingface.co/WenyaoZhang/Hybrid-depth/tree/main). Download the checkpoint folder and set `--load_weights_folder` to its path.

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
  --model_name /path/to/weights \
  --cat_depth_text_logic \
  --use_depth_text_align \
  --n_depth_text_tokens 256
```

Use the same checkpoint as in [Evaluation](#evaluation) (e.g. from [Hugging Face](https://huggingface.co/WenyaoZhang/Hybrid-depth/tree/main)). Outputs are written to the same directory as the input image by default, or set `--vis_dir` to specify an output folder.

---

## Project structure (main parts)

| Path | Description |
|------|-------------|
| `Stage2/` | Monodepth2-style training & evaluation (DINO/CLIP encoder, depth decoder) |
| `manydepth/` | ManyDepth multi-frame training & evaluation |
| `modules/` | Shared modules (e.g. DepthCLIP, MainRunnerLM) |
| `datasets/` | Data loaders and split files |
| `params/` | YAML configs for `main.py` (optional pipeline) |

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
