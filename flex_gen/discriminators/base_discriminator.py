from torch import nn
from torch.nn import functional as F
import torch
from torch.autograd import grad as torch_grad
from einops import rearrange, repeat
from torch import autocast


def hinge_discr_loss(fake_logits, real_logits):
    return (F.relu(1 - real_logits) + F.relu(1 + fake_logits)).mean() / 2


def hinge_gen_loss(fake):
    return -fake.mean()


def vanilla_discr_loss(fake_logits, real_logits):
    return (F.softplus(-real_logits) + F.softplus(fake_logits)).mean() / 2


def vanilla_gen_loss(fake):
    return F.softplus(-fake).mean()


def _sigmoid_cross_entropy_with_logits(labels, logits):
    """
    non-saturating loss
    """
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    condition = logits >= zeros
    relu_logits = torch.where(condition, logits, zeros)
    neg_abs_logits = torch.where(condition, -logits, logits)
    return relu_logits - logits * labels + torch.log1p(torch.exp(neg_abs_logits))


def non_saturate_gen_loss(logits_fake):
    """
    logits_fake: [B 1 H W]
    """
    # B, _, _, _ = logits_fake.shape
    B = logits_fake.shape[0]
    logits_fake = logits_fake.reshape(B, -1)
    logits_fake = torch.mean(logits_fake, dim=-1)
    gen_loss = torch.mean(
        _sigmoid_cross_entropy_with_logits(
            labels=torch.ones_like(logits_fake), logits=logits_fake
        )
    )

    return gen_loss


def non_saturate_discriminator_loss(logits_real, logits_fake):
    # B, _, _, _ = logits_fake.shape
    B = logits_fake.shape[0]
    logits_real = logits_real.reshape(B, -1)
    logits_fake = logits_fake.reshape(B, -1)
    logits_fake = logits_fake.mean(dim=-1)
    logits_real = logits_real.mean(dim=-1)

    real_loss = _sigmoid_cross_entropy_with_logits(
        labels=torch.ones_like(logits_real), logits=logits_real
    )

    fake_loss = _sigmoid_cross_entropy_with_logits(
        labels=torch.zeros_like(logits_fake), logits=logits_fake
    )

    discr_loss = (real_loss + fake_loss).mean()
    return discr_loss / 2


class LeCAM_EMA(object):
    def __init__(self, init=0.0, decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(
            logits_real
        ).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(
            logits_fake
        ).item() * (1 - self.decay)


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + torch.mean(
        F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2)
    )
    return reg


def gradient_penalty(images, output, use_WGAN=False):
    with autocast(enabled=False, device_type=images.device.type):
        gradients = torch_grad(
            outputs=output,
            inputs=images,
            grad_outputs=torch.ones(output.size(), device=images.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = rearrange(gradients, "b ... -> b (...)")
        if use_WGAN:
            return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        else:
            return (gradients.norm(2, dim=1) ** 2).mean()


def grad_layer_wrt_loss(loss: torch.Tensor, layer: nn.Parameter):
    with autocast(enabled=False, device_type=loss.device.type):
        return torch_grad(
            outputs=loss,
            inputs=layer,
            grad_outputs=torch.ones_like(loss),
            retain_graph=True,
        )[0].detach()


def get_perceptual_model(perceptual_model="vgg16"):
    if perceptual_model == "vgg16":
        from flex_gen.layers.perceptual_loss.lpips import LPIPS

        return LPIPS().eval()
    elif perceptual_model == "dino":
        from flex_gen.layers.perceptual_loss.dino_lpips import DINO_LPIPS

        return DINO_LPIPS().eval()
    elif perceptual_model == "resnet":
        from flex_gen.layers.perceptual_loss.restnet_lpips import ResNet_LPIPS

        return ResNet_LPIPS().eval()
    elif perceptual_model == "vgg_logit":
        from flex_gen.layers.perceptual_loss.vgg_lpips import VGG_LPIPS

        return VGG_LPIPS().eval()

    else:
        return None


class BaseDiscriminator(nn.Module):
    disciminator_network: nn.Module = nn.Identity()
    default_params = {
        "loss_type": "hinge",
        "gen_loss_type": None,
        "disc_loss_type": None,
        # "with_gradient_penalty": False,
        "lambda_perceptual_loss": 1.0,
        "lambda_disc_loss": 1.0,
        "lambda_adversarial_loss": 1.0,
        "lambda_grad_penalty": 10.0,
        "lambda_lecam": 0.0,
        "adaptive_weight_max": 1e3,
        "perceptual_model": "vgg16",
        "disable_adaptive_weight": False,
    }

    def __init__(self, **kwargs):
        super(BaseDiscriminator, self).__init__()
        # Warp gan/disc loss type
        for key, value in self.default_params.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.loss_type is None:
            self.loss_type = "hinge"
        if self.gen_loss_type is None:
            self.gen_loss_type = self.loss_type
        if self.disc_loss_type is None:
            self.disc_loss_type = self.loss_type

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)
        self.perceptual_model = get_perceptual_model(self.perceptual_model)

        if self.lambda_lecam > 0:
            self.lecam_ema = LeCAM_EMA()
        else:
            self.lecam_ema = None

    def train(self, mode: bool = True):
        # This can make sure that the perceptual model is always in eval mode
        self.training = mode
        for module in self.children():
            module.train(mode)
        if self.perceptual_model is not None:
            self.perceptual_model.train(False)
        return self

    def eval(self):
        return self.train(False)

    def compute_disc_loss(self, real_logits, fake_logits):
        if self.disc_loss_type == "hinge":
            return hinge_discr_loss(fake_logits, real_logits)
        elif self.disc_loss_type == "vanilla":
            return vanilla_discr_loss(fake_logits, real_logits)
        elif self.disc_loss_type == "non_saturate":
            return non_saturate_discriminator_loss(real_logits, fake_logits)
        else:
            raise NotImplementedError

    def comput_gen_loss(self, fake_logits):
        if self.gen_loss_type == "hinge":
            return hinge_gen_loss(fake_logits)
        elif self.gen_loss_type == "vanilla":
            return vanilla_gen_loss(fake_logits)
        elif self.gen_loss_type == "non_saturate":
            return non_saturate_gen_loss(fake_logits)
        else:
            raise NotImplementedError

    def forward(self, func="get_disc_loss", **kwargs):
        # to avoid DDP error
        if func == "get_disc_loss":
            return self.get_disc_loss(**kwargs)
        elif func == "get_gan_loss":
            return self.get_gan_loss(**kwargs)

    def disc_forward(self, x):
        raise NotImplementedError

    def calculate_gradient_penalty(self, real, real_logits, use_GP=False):
        if use_GP:
            gradient_penalty_loss = gradient_penalty(real, real_logits)
        else:
            gradient_penalty_loss = self.zero
        return gradient_penalty_loss

    def check_if_apply_gradient_penalty(
        self, lambda_grad_penalty=None, apply_gradient_penalty=False
    ):
        return (
            # self.with_gradient_penalty
            lambda_grad_penalty > 0
            and apply_gradient_penalty
        )

    def get_disc_loss(
        self,
        real,
        fake,
        lambda_disc_loss=None,
        lambda_grad_penalty=None,
        apply_gradient_penalty=False,
    ):
        lambda_disc_loss = lambda_disc_loss or self.lambda_disc_loss
        lambda_grad_penalty = lambda_grad_penalty or self.lambda_grad_penalty

        if self.check_if_apply_gradient_penalty(
            lambda_grad_penalty, apply_gradient_penalty
        ):
            real = real.requires_grad_()
            use_GP = True
        else:
            use_GP = False

        real_logits = self.disc_forward(real)
        fake_logits = self.disc_forward(fake.detach())
        discr_loss = self.compute_disc_loss(real_logits, fake_logits)
        if self.lecam_ema is not None and self.lambda_lecam > 0:
            self.lecam_ema.update(real_logits, fake_logits)
            lecam_loss = lecam_reg(real_logits, fake_logits, self.lecam_ema)
        else:
            lecam_loss = self.zero

        gradient_penalty_loss = self.calculate_gradient_penalty(
            real, real_logits, use_GP
        )

        loss_sum = (
            discr_loss * lambda_disc_loss
            + gradient_penalty_loss * lambda_grad_penalty
            + lecam_loss * self.lambda_lecam
        )

        return loss_sum, {
            "discr_loss": discr_loss,
            "gradient_penalty_loss": gradient_penalty_loss,
            "lambda_disc_loss": lambda_disc_loss,
            "lambda_grad_penalty": lambda_grad_penalty,
            "lecam_loss": lecam_loss,
            "disc_loss_sum": loss_sum,
        }

    def get_adaptive_weight(self, gen_loss, perceptual_loss=None, last_dec_layer=None):
        if perceptual_loss != None:
            norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(
                perceptual_loss, last_dec_layer
            ).norm(p=2)

            norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(
                p=2
            )
            adaptive_weight = (
                norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min=1e-3)
            )
            adaptive_weight.clamp_(max=self.adaptive_weight_max)

            if torch.isnan(adaptive_weight).any():
                adaptive_weight = 1.0
        else:
            adaptive_weight = 1.0

        return adaptive_weight

    def get_perceptual_loss(self, real, fake):
        if self.perceptual_model is None:
            return self.zero
        channels = fake.shape[1]

        if channels == 1:
            real = repeat(real, "b 1 h w -> b c h w", c=3)
            fake = repeat(fake, "b 1 h w -> b c h w", c=3)

        return self.perceptual_model(real, fake).mean()

    def get_gan_loss(
        self,
        real,
        fake,
        rec_loss=None,
        last_dec_layer=None,
        lambda_adversarial_loss=None,
        lambda_perceptual_loss=None,
    ):

        # notice that lambda_adversarial_loss could be 0.0
        if lambda_adversarial_loss is None:
            lambda_adversarial_loss = self.lambda_adversarial_loss

        lambda_perceptual_loss = lambda_perceptual_loss or self.lambda_perceptual_loss

        if lambda_perceptual_loss is None or lambda_perceptual_loss == 0.0:
            perceptual_loss = self.zero
        else:
            perceptual_loss = self.get_perceptual_loss(real, fake)

        if (
            lambda_adversarial_loss is None
            or lambda_perceptual_loss is None
            or lambda_adversarial_loss == 0.0
            or rec_loss is None
        ):
            gen_loss = self.zero
            adaptive_weight = 1.0
        else:
            fake_logits = self.disc_forward(fake)
            gen_loss = self.comput_gen_loss(fake_logits)

            if self.disable_adaptive_weight:
                adaptive_weight = 1.0
            else:
                adaptive_weight = self.get_adaptive_weight(
                    gen_loss,
                    perceptual_loss * lambda_perceptual_loss + rec_loss,
                    last_dec_layer,
                )
        loss_sum = (
            perceptual_loss * lambda_perceptual_loss
            + gen_loss * adaptive_weight * lambda_adversarial_loss
        )

        return loss_sum, {
            "perceptual_loss": perceptual_loss.item(),
            "gen_loss": gen_loss.item(),
            "adaptive_weight": adaptive_weight,
            "lambda_perceptual_loss": lambda_perceptual_loss,
            "lambda_adversarial_loss": lambda_adversarial_loss,
            "gan_loss_sum": loss_sum.item(),
        }
