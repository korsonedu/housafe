#!/bin/bash
# AutoDL GPU 环境初始化脚本
# 在 AutoDL JupyterLab Terminal 中运行: bash setup_gpu.sh

set -e

echo "=== 安装依赖 ==="
pip install numpy scipy scikit-learn tqdm -q

echo ""
echo "=== 验证 PyTorch CUDA ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'CUDA version: {torch.version.cuda}')
"

echo ""
echo "=== 测试 Encoder Forward Pass ==="
python -c "
import sys
sys.path.insert(0, '.')
from ai.encoder.encoder_model import EncoderModel
import torch
model = EncoderModel()
seq = torch.randn(4, 32, 64, 5)
out = model(seq)
print(f'S: {out[\"S\"].shape} | z: {out[\"z\"].shape} | logits: {out[\"logits\"].shape}')
total = sum(p.numel() for p in model.parameters())
print(f'Total params: {total:,}')
print('Forward pass OK')
"

echo ""
echo "=== 环境就绪 ==="
echo "运行训练: python ai/encoder/train_encoder.py --data ./3dpchm_frames.npz --device cuda --epochs 300"
