# DFSTrack

`DFSTrack` is a clean visual-language tracking baseline derived from the original repository structure.

## What is kept

- Fast-iTPN image encoder
- RoBERTa text encoder
- Center head tracking pipeline

## What is removed

- Temporal memory and dynamic state modules
- Confidence prediction branch
- Subject-mask prediction branch
- Extra ATC-specific fusion components

## Training

Single GPU:

```bash
cd /home/ssa/code/ATCTrack
python tracking/train.py --script dfstrack --config dfstrack_base --save_dir ./output --mode single
```

Multiple GPUs:

```bash
cd /home/ssa/code/ATCTrack
python tracking/train.py --script dfstrack --config dfstrack_base --save_dir ./output --mode multiple --nproc_per_node 3
```

## Testing

```bash
cd /home/ssa/code/ATCTrack
python tracking/test.py --tracker_name dfstrack --tracker_param dfstrack_base --dataset_name tnl2k --ckpt_path ./output/checkpoints/train/dfstrack/dfstrack_base/DFSTrack_ep0180.pth.tar
```

## Notes

- `DFSTrack` first looks for a local Hugging Face style RoBERTa folder at `resource/pretrained_models/roberta-base`.
- If that folder is missing, it falls back to `roberta-base` from Hugging Face.















--------------------------------------------------------------------------------
### Git有两种主要的通信协议
其中HTTPS使用用户名和密码（现已改为个人访问令牌），所以会遇见fatal: 'https://github.com/ssaishandsome/dfstrack.git/' 鉴权失败的问题
而SSH使用的公钥密钥对，无需每次输入密码（配置后）

HTTPS：使用443端口，基于HTTP/TLS加密
SSH：使用22端口，基于SSH协议加密