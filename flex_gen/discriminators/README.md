

In general speaking, the custom discriminators should follows:

```python
    class SomeDiscriminator(BaseDiscriminator):
        def __init__(self, a, b , **kwargs):
            super().__init__(**kwargs)
            self.blocks = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
            )


        def disc_forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward through the discriminator."""
            return self.blocks(x)
```