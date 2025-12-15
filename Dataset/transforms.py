#!/usr/bin/env python3


from torchvision import transforms as T

# default mean and std values for vit
# mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#TODO: check mean and std for MAE and MoCo
default_norm_data = {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }

"""Image transformations for FGVC datasets"""
# https://github.com/KMnP/vpt/blob/main/src/data/transforms.py
def get_transforms(args, split, target_size, pretrained_cfg=None):
    
    if pretrained_cfg is not None:
        norm_data = {
                'mean': pretrained_cfg['mean'],
                'std': pretrained_cfg['std']
            }
    else:
        norm_data = default_norm_data 
    
    if target_size == 448:
        resize_dim = 512
    elif target_size == 224:
        resize_dim = 256
    elif target_size == 384:
        resize_dim = 438
        
    if split == "train":
        transform = T.Compose(
            [
                T.RandomResizedCrop(target_size, interpolation=3),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize(**norm_data)
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize(resize_dim, interpolation=3),
                T.CenterCrop(target_size),
                T.ToTensor(),
                T.Normalize(**norm_data)
            ]
        )
    return transform


"""Image transformations for VTAB datasets"""
def get_vtab_transforms(args, split, target_size, pretrained_cfg=None):
    
    if pretrained_cfg is not None:
        norm_data = {
                'mean': pretrained_cfg['mean'],
                'std': pretrained_cfg['std']
            }
    else:
        norm_data = default_norm_data 
    
    if split == "train":
        transforms = T.Compose([
            T.Resize(target_size, interpolation=3),
            T.CenterCrop(target_size),
            T.ToTensor(),
            T.Normalize(**norm_data)
        ])
    else:
        transforms = T.Compose([
            T.Resize(target_size, interpolation=3),
            T.CenterCrop(target_size),
            T.ToTensor(),
            T.Normalize(**norm_data)
        ])
    return transforms