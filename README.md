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
- **NYU Depth v2**: Set `--dataset nyu` and point `--data_path` to your NYU path.

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

```bash
cd Stage2
python evaluate_depth.py \
  --load_weights_folder /path/to/weights \
  --eval_split eigen \
  --eval_mono
```

For full benchmark evaluation, use `--eval_split benchmark` and the corresponding split files.

### Single-image inference (Stage2)

```bash
cd Stage2
python test_simple.py --image_path /path/to/image --model_path /path/to/weights
```

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
