
We expect to initialize the model in the following way:
```python

def __init__(self,  **kwargs):
    """
    input_shape: tuple, shape of the input image
    latent_dim: int, dimension of the latent space
    """
    super().__init__(**self.capture_init_args(locals()))
    raise NotImplementedError

```


We expect all models have the following functions :

```python
def toeknize(self, x):
    """
    return returns a dictionary, includes 'codes'
    """
    raise NotImplementedError

def encode(self, x):
    """
    return returns a dictionary, includes 'quantized'
    """
    raise NotImplementedError

def decode(self, x):
    """
    return is the reconstracted image
    """
    raise NotImplementedError

def forward(self, x, return_codes=False):
    """
    return is a 2-tuple of (loss_sum, loss_breakdown)
    loss_breakdown = {
        "recon": x_recon,
        "recon_loss": recon_loss,
        "aux_loss": aux_losses,
        "quantized": z,
        "codes": codes,
        "random_latent": encoded_dist.sample(),
    }
    """
    raise NotImplementedError

def get_last_dec_layer(self):
    raise NotImplementedError

```