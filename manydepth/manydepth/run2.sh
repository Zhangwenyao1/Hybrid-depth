# export PATH="/data/ericliu/my_packages/miniconda3/bin:$PATH"
# __conda_setup="$('/data/ericliu/my_packages/miniconda3/condabin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# eval "$__conda_setup"
# cd /code/CFMDE-main/manydepth
conda activate cfmde
python manydepth/train.py --log_dir manydepth/exps --use_cfmde --backbone_model dino_clip --use_dpt_teacher --model_name dc_pretrained_lowlr --stage1_checkpoint_path "/model/ericliu/CFMDE-pretrained/epoch=19.ckpt" --learning_rate 1e-5
