# Download SmoothNet Checkpoints

The SmoothNet checkpoints need to be downloaded from:

**Google Drive:** https://drive.google.com/drive/folders/19Cu-_gqylFZAOTmHXzK52C80DKb0Tfx_?usp=sharing

**Baidu Netdisk:** https://pan.baidu.com/s/1J6EV4uwThcn-W_GNuc4ZPw?pwd=eb5x

## Required Checkpoint for PARE Integration

For the PARE pipeline, download:
- `pw3d_spin_3D/checkpoint_32.pth.tar` (recommended, window_size=32)
- Or any of: `checkpoint_8.pth.tar`, `checkpoint_16.pth.tar`, `checkpoint_32.pth.tar`, `checkpoint_64.pth.tar`

## Installation Steps

1. Download the checkpoint file(s) from Google Drive
2. Place them in: `smoothnet/data/checkpoints/pw3d_spin_3D/`
3. Make sure the filename matches what you specify in `--smoothnet_checkpoint`

## Alternative Checkpoints

You can also use checkpoints trained on other datasets:
- `aist_vibe_3D/checkpoint_32.pth.tar`
- `h36m_fcn_3D/checkpoint_32.pth.tar`

Just update the `--smoothnet_checkpoint` path accordingly.
