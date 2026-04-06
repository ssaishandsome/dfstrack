#!/bin/bash

# export PATH=/usr/local/cuda/bin:$PATH
# export CUDA_HOME=/usr/local/cuda

unset LD_LIBRARY_PATH # 清空动态链接库，不让去其他地方寻找，使用conda环境中的lib即可
export LD_LIBRARY_PATH=/miniconda3/envs/savlt/lib:$LD_LIBRARY_PATH

source /root/miniconda3/etc/profile.d/conda.sh
conda activate savlt

config='joint_8_noisyTemp1_v3'
python -m torch.distributed.launch --nproc_per_node 3 lib/train/run_training.py --script baseline_clip_cocoop_joint_audio  --config ${config} --save_dir /18009672469/codes/SAVLT-Speech/runs/audio

