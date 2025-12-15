import torchvision.datasets as datasets
from .transforms import *

_CATALOG = {
    "tv_cifar100": "CIFAR100",
    "tv_aricraft": "FGVCAircraft",
    "tv_food": "Food101"
}

# create torchvision dataset
def create_tv_dataset(args, quiet=False):
    dataset_name = args.dataset
    root = args.data_dir
    try:
        if dataset_name == "tv_cifar100":
            train_dataset = datasets.__dict__[_CATALOG[dataset_name]](root=root, train=True, download=True,
                                    transform=get_transforms(args, "train", args.data_cropsize))
            val_dataset = datasets.__dict__[_CATALOG[dataset_name]](root=root, train=False, download=True,
                                    transform=get_transforms("val", args.data_cropsize))
            num_classes = len(train_dataset.classes)
        else:
            train_dataset = datasets.__dict__[_CATALOG[dataset_name]](root=root, split='train', download=True,
                                    transform=get_transforms(args, "train", args.data_cropsize))
            val_dataset = datasets.__dict__[_CATALOG[dataset_name]](root=root, split='test', download=True,
                                    transform=get_transforms(args, "val", args.data_cropsize))
            num_classes = len(train_dataset.classes)
        return train_dataset, val_dataset, num_classes
    except KeyError:
        raise Exception(f"Dataset '{dataset_name}' does not exist in torchvision.")