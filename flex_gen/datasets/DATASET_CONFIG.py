"""
    check https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/transforms_factory.py#L237

    It is very very imporant to use the correct mean and std for each dataset, e.g.
    Dataset         Image Size      Mean                        Std                         Samples
    MNIST,        28x28          0.1307,0.1307,0.1307           0.3081,0.3081,0.3081        60000
    CIFAR10,      32x32          0.4914, 0.4822, 0.4465         0.2470, 0.2435, 0.2616
    CIFAR100,     32x32          0.5071, 0.4867, 0.4408         0.2675, 0.2565, 0.2761
    SVHN,                   0.4377, 0.4438, 0.4728         0.1980, 0.2010, 0.1970
    FASHIONMNIST,           0.2860, 0.2860, 0.2860         0.3205, 0.3205, 0.3205
    IMAGENET,               0.485, 0.456, 0.406            0.229, 0.224, 0.225

    DATASET_KEYS = {
    "cifar10": ["img", "label"],
    "cifar100": ["img", "fine_label"],
    "mnist": ["image", "label"],
}
"""

"""
For diffusion, we should maybe use mean/std 0/1
"""
DATASET_CONFIGS = {
    # "mnist": {
    #     "image_size": 28,
    #     "mean": (0.1307, 0.1307, 0.1307),
    #     "std": (0.3081, 0.3081, 0.3081),
    #     "samples": 60000,
    #     "data_keys": ["image", "label"],
    # },
    # "cifar10": {
    #     "image_size": 32,
    #     "mean": (0.4914, 0.4822, 0.4465),
    #     "std": (0.2470, 0.2435, 0.2616),
    #     "samples": 50000,
    #     "data_keys": ["img", "label"],
    # },
    # "cifar100": {
    #     "image_size": 32,
    #     "mean": (0.5071, 0.4867, 0.4408),
    #     "std": (0.2675, 0.2565, 0.2761),
    #     "samples": 50000,
    #     "data_keys": ["img", "fine_label"],
    # },
    # "imagenet-1k": {
    #     "image_size": 224,
    #     "mean": (0.485, 0.456, 0.406),
    #     "std": (0.229, 0.224, 0.225),
    #     "samples": 1281167,
    #     "data_keys": ["image", "label"],
    # },
    # "celeba-hq": {
    #     "image_size": 256,
    #     "mean": (0.5, 0.5, 0.5),
    #     "std": (1, 1, 1),
    #     "data_keys": ["image", "label"],
    # },
    # "ucf101": {
    #     "image_size": 256,
    #     "mean": (0.5, 0.5, 0.5),
    #     "std": (1, 1, 1),
    #     "data_keys": ["video", "label"],
    # },
    # "faceforensics": {
    #     "image_size": 256,
    #     "mean": (0.5, 0.5, 0.5),
    #     "std": (1, 1, 1),
    #     "data_keys": ["video", "label"],
    # },
}


def query_mean_from_dataset_name(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name]["mean"]
    else:
        # return [0, 0, 0]
        return [0.5, 0.5, 0.5]


def query_std_from_dataset_name(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name]["std"]
    else:
        # return [1, 1, 1]
        return [0.5, 0.5, 0.5]


def query_mean_std_from_dataset_name(dataset_name):
    dataset_name = dataset_name.lower()
    mean, std = query_mean_from_dataset_name(dataset_name), query_std_from_dataset_name(
        dataset_name
    )
    return mean, std
