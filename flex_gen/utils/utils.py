import matplotlib.pyplot as plt
import numpy as np
import torch
import importlib

import torch.utils


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def convert_tensor_to_heatmap(img_tensor):
    # Normalize the image tensor to [0, 1]
    normalized_img = (img_tensor - img_tensor.min()) / (
        img_tensor.max() - img_tensor.min()
    )

    if normalized_img.ndim == 4:
        # If the input is a batch of images, take the first one
        normalized_img = np.stack(
            plt.cm.hot(img.numpy())[..., :3] for img in normalized_img
        )
    else:
        normalized_img = plt.cm.hot(normalized_img.numpy())[..., :3]
    return normalized_img


def rescale_image_tensor(img_tensor, mean, std):

    if isinstance(mean, (int, float)):
        mean = [mean] * 3
    if isinstance(std, (int, float)):
        std = [std] * 3
    mean = torch.tensor(mean).view(3, 1, 1).type_as(img_tensor).unsqueeze(0)
    std = torch.tensor(std).view(3, 1, 1).type_as(img_tensor).unsqueeze(0)

    if img_tensor.ndim == 5:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return img_tensor * std + mean
