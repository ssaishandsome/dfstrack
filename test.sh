#!/bin/bash

# export PATH=/usr/local/cuda/bin:$PATH
# export CUDA_HOME=/usr/local/cuda

cd /home/ssa/code/DFSTrack

source /home/ssa/anaconda3/etc/profile.d/conda.sh
conda activate savlt

python tracking/test.py --tracker_name dfstrack --dataset_name lasot_lang --tracker_param dfstrack_base --ckpt_path /home/ssa/code/DFSTrack/output/checkpoints/train/dftrack/dftrack_base/DFSTrack_ep0174.pth.tar

python tracking/test.py --tracker_name dfstrack --dataset_name tnl2k --tracker_param dfstrack_base --ckpt_path /home/ssa/code/DFSTrack/output/checkpoints/train/dftrack/dftrack_base/DFSTrack_ep0174.pth.tar


# -m torch.distributed.launch --nproc_per_node 1