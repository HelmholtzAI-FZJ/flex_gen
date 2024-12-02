pip install numpy==1.26.4
pip install torch
pip install datasets
pip install accelerate
pip install timm
pip install transformers
pip install torchmetrics
pip install torch_fidelity
pip install diffusers
pip install torchdiffeq
pip install xformers
pip install einops
pip install deepspeed
pip install wandb
pip install flash-attn
pip install omegaconf
pip install matplotlib
pip install beartype
pip install kornia # for magvit2
pip install PyAV
pip install moviepy 
pip install imageio
pip install triton==2.2.0
# FOR ZIGMA & Flash-linear-attention
pip install mamba-ssm
pip install causal_conv1d

## For Flash-linear-attention
git clone https://github.com/sustcsonglin/flash-linear-attention ./resources/flash-linear-attention
pip install -e ./resources/flash-linear-attention

# git clone https://github.com/Doraemonzzz/hgru2-pytorch.git ./resources/hgru2-pytorch
pip install -e ./resources/hgru2-pytorch/