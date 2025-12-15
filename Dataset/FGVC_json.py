#!/usr/bin/env python3

"""JSON dataset: support CUB, NABrids, Flower, Dogs and Cars"""

import os
import json
import torchvision as tv
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
from .transforms import *
from typing import Union

_FGVC_CATALOG = {
    "cub": "fgvc/CUB200",
    'flowers': "fgvc/FLOWERS102",
    'cars': "fgvc/CARS196",
    'dogs': "fgvc/DOGS120",
    "nabirds": "fgvc/NABIRDS555",
}

class FGVCDataset(Dataset):
    def __init__(self, args, split):
        assert split in {
            "train",
            "val",
            "test",
        }, "Split '{}' not supported for {} dataset".format(
            split, args.dataset)
        assert args.dataset in _FGVC_CATALOG.keys(), \
            "Dataset {} is not support".format(args.dataset)
        self.name = _FGVC_CATALOG[args.dataset]   
        self.data_dir = os.path.join(args.data_dir, self.name)
        
        #TODO: check what is this for
        self.data_percentage = 1.0
        
        ##
        self.args = args
        self._split = split
        self._construct_imdb(args)
        self.transform = get_transforms(args, split, args.data_cropsize)

    def read_json(self, filename: str) -> Union[list, dict]:
        """read json files"""
        with open(filename, "rb") as fin:
            data = json.load(fin)
        return data
    
    def get_anno(self):
        anno_path = os.path.join(self.data_dir, "{}.json".format(self._split))
        if "train" in self._split:
            if self.data_percentage < 1.0:
                anno_path = os.path.join(
                    self.data_dir,
                    "{}_{}.json".format(self._split, self.data_percentage)
                )
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)
        return self.read_json(anno_path)

    def get_imagedir(self):
        assert os.path.exists(self.data_dir), "{} dir not found".format(self.data_dir)
        if self.args.dataset == "flowers" or self.args.dataset == "cars":
            return self.data_dir
        else:
            return os.path.join(self.data_dir, "images")
        
    def _construct_imdb(self, args):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        
        # Construct the image db
        self._imdb = []
        self._dict = {}
        counter = 0
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            im_path = os.path.join(img_dir, img_name)
            self._imdb.append({"im_path": im_path, "class": cont_id})
            # create image dict
            if cls_id not in self._dict:
                self._dict[cls_id] = []
            self._dict[cls_id].append([im_path, counter])
            counter += 1
        
        self.num_imgs = len(self._imdb)
        self.num_classes = len(self._class_ids)
        
    def get_info(self):
        return self.get_num_imgs(), self.get_num_class()

    def get_num_class(self):
        return len(self._class_ids)

    def get_img_dict(self):
        return self._dict

    def get_num_imgs(self):
        return self.num_imgs
    
    def __getitem__(self, index):
        # Load the image
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])        
        label = self._imdb[index]["class"]
        im = self.transform(im)
        if self._split == "train":
            index = index
        else:
            index = f"{self._split}{index}"
        return im, label

    def __len__(self):
        return len(self._imdb)

def create_fgvc_dataset(args, quiet=False):
    train_data = FGVCDataset(args, "train")
    val_data = FGVCDataset(args, "val")
    test_data = FGVCDataset(args, "test")
    
    if not quiet:
        print("Constructing FGVC: {}".format(args.dataset))
        print("[{}] S/CLS Train: {}/{} | Val: {}/{} | Test: {}/{}".format(
            train_data.name, 
            train_data.get_info()[0], train_data.get_info()[1], 
            val_data.get_info()[0], val_data.get_info()[1], 
            test_data.get_info()[0], test_data.get_info()[1]
        ))
    
    return train_data, val_data, test_data, \
        train_data.get_num_class()
    
#warning: must be called with: python -m Dataset.FGVC_json
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cub')
    parser.add_argument("--data_dir", type=str, default="./vpt_data", help="dataset path")
    parser.add_argument("--data_cropsize", type=int, default=224, help="crop size")
    args = parser.parse_args()
    train_data, val_data, test_data, num_class = create_fgvc_dataset(args)
    print("Success")
