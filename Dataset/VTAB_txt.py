from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from .transforms import *
import os


class VTABDataset(Dataset):
    def __init__(self, args, root_dir, split_txt, split='train'):
        """
        root_dir: dataset path
        split_txt: dataset split, train or test.
        """
        self.image_list = []
        self.id_list = []
        self.name = args.dataset
        self.root_dir = root_dir
        self.transform = get_vtab_transforms(args, split, args.data_cropsize)
        
        with open(split_txt, 'r') as f:
            line = f.readline()
            while line:
                img_name = line.split()[0]
                label = int(line.split()[1])
                self.image_list.append(os.path.join(self.root_dir, img_name))
                self.id_list.append(label)
                line = f.readline()
                
        # create img dict
        self._dict = {}
        for i, label in enumerate(self.id_list):
            if label not in self._dict:
                self._dict[label] = []
            self._dict[label].append([self.image_list[i], i])        
        
        #self.num_classes = max(self.id_list) + 1
        self.num_classes = len(set(self.id_list))
        self.num_imgs = len(self.id_list)
    
    def get_img_dict(self):
        return self._dict
    
    def get_num_class(self):
        return self.num_classes
    
    def get_num_imgs(self):
        return self.num_imgs

    def get_info(self):
        return self.get_num_imgs(), self.get_num_class()
    
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label = self.id_list[idx]
        #img_name = os.path.join(self.root_dir, img_name)
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, label


_VTAB_CATALOG = {
    "caltech101": "vtab-1k/caltech101",
    "cifar100": "vtab-1k/cifar",
    "clevr_count": "vtab-1k/clevr_count",
    "clevr_dist": "vtab-1k/clevr_dist",
    "retinopathy": "vtab-1k/diabetic_retinopathy",
    "dmlab": "vtab-1k/dmlab",
    "dsprites_loc": "vtab-1k/dsprites_loc",
    "dsprites_ori": "vtab-1k/dsprites_ori",
    "dtd": "vtab-1k/dtd",
    "eurosat": "vtab-1k/eurosat",
    "kitti": "vtab-1k/kitti",
    "oxford_flowers102": "vtab-1k/oxford_flowers102",
    "oxford_iiit_pet": "vtab-1k/oxford_iiit_pet",
    "patch_camelyon": "vtab-1k/patch_camelyon",
    "resisc45": "vtab-1k/resisc45",
    "smallnorb_azi": "vtab-1k/smallnorb_azi",
    "smallnorb_ele": "vtab-1k/smallnorb_ele",
    "sun397": "vtab-1k/sun397",
    "svhn": "vtab-1k/svhn"
}

def create_vtab_val_dataset(args, quiet=False):
    root_dir = os.path.join(args.data_dir, _VTAB_CATALOG[args.dataset])
    train_txt = os.path.join(root_dir, "train800.txt")
    val_txt = os.path.join(root_dir, "val200.txt")
    train_data = VTABDataset(args, root_dir, train_txt, 'train')
    val_data = VTABDataset(args, root_dir, val_txt, 'test')
    
    if not quiet:
        print("\nConstructing VTAB: {}".format(args.dataset))
        print("[{}] sample/cls Train: {}/{} | Val: {}/{}".format(
            train_data.name,
            train_data.get_info()[0], train_data.get_info()[1], 
            val_data.get_info()[0], val_data.get_info()[1]
        ))
    
    return train_data, val_txt, max(train_data.num_classes, val_data.num_classes)
    
    
def create_vtab_dataset(args, quiet=False):
    root_dir = os.path.join(args.data_dir, _VTAB_CATALOG[args.dataset])
    train_txt = os.path.join(root_dir, "train800val200.txt")
    test_txt = os.path.join(root_dir, "test.txt")
    train_data = VTABDataset(args, root_dir, train_txt, 'train')
    test_data = VTABDataset(args, root_dir, test_txt, 'test')
    
    if not quiet:
        print("Constructing VTAB: {}".format(args.dataset))
        print("[{}] S/CLS Train: {}/{} | Test: {}/{}".format(
            train_data.name,
            train_data.get_info()[0], train_data.get_info()[1], 
            test_data.get_info()[0], test_data.get_info()[1]
        ))
    
    return train_data, test_data, max(train_data.num_classes, test_data.num_classes)


#warning must be called with: python -m Dataset.VTAB_txt
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='sun397')
    parser.add_argument("--data_dir", type=str, default="./vpt_data", help="dataset path")
    parser.add_argument("--data_cropsize", type=int, default=224, help="crop size")
    args = parser.parse_args()
    train_data, val_data, num_class = create_vtab_val_dataset(args)
    train_data, test_data, num_class = create_vtab_dataset(args)
    print("Success")
    
