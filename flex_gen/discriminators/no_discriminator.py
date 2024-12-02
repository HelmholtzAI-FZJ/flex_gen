from .base_discriminator import BaseDiscriminator


class NoDiscriminator(BaseDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.disciminator_network = None

    def disc_forward(self, x):
        return x

    def get_disc_loss(
        self,
        real,
        fake,
        lambda_disc_loss=None,
        lambda_grad_penalty=None,
        apply_gradient_penalty=False,
    ):
        loss = self.zero * 0
        return loss, {
            "discr_loss": self.zero,
            "gradient_penalty_loss": self.zero,
            "lambda_disc_loss": 0,
            "lambda_grad_penalty": 0,
            "disc_loss_sum": self.zero,
        }

    def get_gan_loss(
        self,
        real,
        fake,
        lambda_perceptual_loss=None,
        rec_loss=None,
        **kwargs,
    ):
        lambda_perceptual_loss = lambda_perceptual_loss or self.lambda_perceptual_loss
        if lambda_perceptual_loss is None or lambda_perceptual_loss == 0.0:
            perceptual_loss = self.zero
        else:
            perceptual_loss = self.get_perceptual_loss(real, fake)

        loss = perceptual_loss * lambda_perceptual_loss + self.zero
        return loss, {
            "perceptual_loss": perceptual_loss.item(),
            "gen_loss": self.zero.item(),
            "adaptive_weight": 0,
            "lambda_perceptual_loss": lambda_perceptual_loss,
            "lambda_adversarial_loss": 0,
            "gan_loss_sum": loss.item(),
        }
