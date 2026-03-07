export PATH="/data/ericliu/my_packages/miniconda3/bin:$PATH"
__conda_setup="$('/data/ericliu/my_packages/miniconda3/condabin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
cd /code/CFMDE-main/manydepth
conda activate cfmde
python manydepth/train.py --log_dir manydepth/exps --use_cfmde --backbone_model dino_clip --use_dpt_teacher --model_name dc_pre_text_freeze_teacher_lowlr --stage1_checkpoint_path "/model/ericliu/CFMDE-pretrained/epoch=19.ckpt" --learning_rate 1e-5 --mono_weights_folder /code/CFMDE-main/manydepth/manydepth/exps/dinoclip_dptteacher_lowlr_fixed/models/weights_19 --freeze_teacher_and_pose

#  --use_depth_text_align --cat_depth_text_logic
