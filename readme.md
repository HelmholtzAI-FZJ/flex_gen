# **Scaling Image Tokenizers with Grouped Spherical Quantization**
---

[Paper link](https://arxiv.org/abs/2412.02632) | [HF Checkpoints](https://huggingface.co/collections/HelmholtzAI-FZJ/grouped-spherical-quantization-674d6f9f548e472d0eaf179e)

In [GSQ](https://arxiv.org/abs/2412.02632), we show the optimized training hyper-parameters and configs for quantization based image tokenizer. We also show how to scale the latent, vocab size etc. appropriately to achieve better reconstruction performance. 

![dim-vocab-scaling.png](./figures/dim-vocab-scaling.png)

We also show how to scaling the latent (and group) appropriately when pursuing high down-sample ratio in compression. 

![spatial_scale.png](./figures//spatial_scale.png)

The group scaling experiment of GSQ:

---
| **Models**                           | \( G $\times$ d \)    | **rFID ↓** | **IS ↑** | **LPIPS ↓** | **PSNR ↑** | **SSIM ↑** | **Usage ↑** | **PPL ↑**   |
|--------------------------------------|---------------------|------------|----------|-------------|------------|------------|-------------|-------------|
| **GSQ F8-D64** \( V=8K \)    | \( 1 $\times$ 64 \)   | 0.63       | 205      | 0.08        | 22.95      | 0.67       | 99.87%      | 8,055       |
|                                      | \( 2 $\times$ 32 \)   | 0.32       | 220      | 0.05        | 25.42      | 0.76       | 100%        | 8,157       |
|                                      | \( 4 $\times$ 16 \)   | 0.18       | 226      | 0.03        | 28.02      | 0.83       | 100%        | 8,143       |
|                                      | \( 16 $\times$ 4 \)   | **0.03**   | **233**  | **0.004**   | **34.61**  | **0.91**   | **99.98%**  | **6,775**   |
| **GSQ F16-D16**  \( V=256K \) | \( 1 $\times$ 16 \)   | 1.63       | 179      | 0.13        | 20.70      | 0.56       | 100%        | 254,044     |
|                                      | \( 2 $\times$ 8 \)    | 0.82       | 199      | 0.09        | 22.20      | 0.63       | 100%        | 257,273     |
|                                      | \( 4 $\times$ 4 \)    | 0.74       | 202      | 0.08        | 22.75      | 0.63       | 62.46%      | 43,767      |
|                                      | \( 8 $\times$ 2 \)    | 0.50       | 211      | 0.06        | 23.62      | 0.66       | 46.83%      | 22,181      |
|                                      | \( 16 $\times$ 1 \)   | 0.52       | 210      | 0.06        | 23.54      | 0.66       | 0.508%      | 181         |
|                                      | \( 16 $\times$ 1^* \) | 0.51       | 210      | 0.06        | 23.52      | 0.66       | 0.526%      | 748         |
| **GSQ F32-D32** \( V=256K \) | \( 1 $\times$ 32 \)   | 6.84       | 95       | 0.24        | 17.83      | 0.40       | 100%        | 245,715     |
|                                      | \( 2 $\times$ 16 \)   | 3.31       | 139      | 0.18        | 19.01      | 0.47       | 100%        | 253,369     |
|                                      | \( 4 $\times$ 8 \)    | 1.77       | 173      | 0.13        | 20.60      | 0.53       | 100%        | 253,199     |
|                                      | \( 8 $\times$ 4 \)    | 1.67       | 176      | 0.12        | 20.88      | 0.54       | 59%         | 40,307      |
|                                      | \( 16 $\times$ 2 \)   | 1.13       | 190      | 0.10        | 21.73      | 0.57       | 46%         | 30,302      |
|                                      | \( 32 $\times$ 1 \)   | 1.21       | 187      | 0.10        | 21.64      | 0.57       | 0.54%         | 247         |
---


## Use Pre-trained GSQ-Tokenizer

```python
from flex_gen import autoencoders
from timm import create_model

# ============= From HF's repo
model=create_model('flexTokenizer', pretrained=True,
                   repo_id='HelmholtzAI-FZJ/GSQ-F8-D8-V64k',)
									 
# ============= From Local Checkpoint
model=create_model('flexTokenizer', pretrained=True,
                   path='PATH/your_checkpoint.pt', )
```

---

## Training your tokenizer

### Set-up Python Virtual Environment

```python
sh gen_env/setup.sh

source ./gen_env/activate.sh

#! This will run pip install to download all required lib
sh ./gen_env/install_requirements.sh 

```

### Run Training

```python
# Single GPU
python -W ignore ./scripts/train_autoencoder.py 

# Multi GPU
torchrun --nnodes=1 --nproc_per_node=4 ./scripts/train_autoencoder.py --config-file=PATH/config_name.yaml \
--output_dir=./logs_test/test opts train.num_train_steps=100 train_batch_size=16
```

### Run Evaluation

Add the checkpoint path that your want to test in `evaluation/run_tokenizer_eval.sh`

```bash
# For example
...
configs_of_training_lists=()
configs_of_training_lists=("logs_test/test/")
...
```

And run `sh evaluation/run_tokenizer_eval.sh` it will automatically scan `folder/model/eval_xxx.pth` for tokenizer evaluation

---

# **Citation**

```bash
@misc{GSQ,
      title={Scaling Image Tokenizers with Grouped Spherical Quantization}, 
      author={Jiangtao Wang and Zhen Qin and Yifan Zhang and Vincent Tao Hu and Björn Ommer and Rania Briq and Stefan Kesselheim},
      year={2024},
      eprint={2412.02632},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.02632}, 
}
```
