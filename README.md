# Hybrid-depth

Monocular depth estimation with DINO/CLIP-driven representations.

⭐ If this project helps you, please give us a star on GitHub!

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

If you use this code in your work, please cite:

```bibtex
@misc{hybriddepth,
  author = {Zhangwenyao},
  title  = {Hybrid-depth: Monocular Depth Estimation with DINO/CLIP},
  year   = {2025},
  url    = {https://github.com/Zhangwenyao1/Hybrid-depth}
}
```
