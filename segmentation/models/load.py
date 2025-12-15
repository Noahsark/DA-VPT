
import numpy as np
import warnings
from typing import List, Optional
import os.path as osp

import mmengine
import mmengine.fileio as fileio
from mmengine.fileio import join_path, list_from_file, load
from mmengine.utils import is_abs
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.datasets.transforms import LoadAnnotations, PackSegInputs
from mmseg.datasets import ADE20KDataset, PascalContextDataset
from mmseg.structures import SegDataSample
from mmseg.registry import TRANSFORMS, DATASETS

@DATASETS.register_module()
class ADE20KDatasetWithScene(ADE20KDataset):
    def __init__(self, label_type='obj', label_file=None,
                 **kwargs):
        data_root = kwargs.get('data_root', None)
        self.label_type = label_type
        self.label_file = label_file
        if self.label_file and not is_abs(self.label_file) and data_root:
            self.label_file = join_path(data_root, self.label_file)
        
        super().__init__(**kwargs)
        
    def _str_to_label(self, label_str: str) -> int:
        """Convert label string to label id."""
        if label_str not in self.label_map:
            self.label_map[label_str] = len(self.label_map)
        return self.label_map[label_str]
    
    def _list_label_filter(self, label_list: List[str], filtered_val=-1) -> List[int]:
        """Filter the label list."""
        return [int(z) for z in label_list if z != str(filtered_val)]
    
    
    #! this is run in super().__init__()
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        self.label_map = dict()
        
        # load label_file
        ann_info = dict()
        if not osp.isdir(self.label_file) and self.label_file:
            assert osp.isfile(self.label_file), \
                 f'Failed to load `label_file` {self.label_file}'
            lines = mmengine.list_from_file(
                 self.label_file, backend_args=self.backend_args)
            if self.label_type == 'scene':
                for line in lines:
                    img_name = line.split(' ')[0]
                    img_path = osp.join(img_dir, img_name + self.img_suffix)
                    ann_info[img_path] = self._str_to_label(line.split(' ')[1])
            elif self.label_type == 'obj':
                for line in lines:
                    img_name = line.split(' ')[0].split('.')[0]
                    img_path = osp.join(img_dir, img_name + self.img_suffix)
                    ann_info[img_path] = self._list_label_filter(
                        line.split(' ')[1:])
            else:
                raise ValueError('label_type should be either "scene" or "obj"')
            
        else:
            raise ValueError('label_file should be a file path')
        
        # num_classes = len(self.label_map)
        
        _suffix_len = len(self.img_suffix)
        for img in fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args):
            img_path = osp.join(img_dir, img)
            data_info = dict(img_path=img_path)
            if ann_dir is not None:
                seg_map = img[:-_suffix_len] + self.seg_map_suffix
                data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
            data_info['label_map'] = None # get_label_map(new_classes)
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['scene_label'] = ann_info[img_path]
            data_info['seg_fields'] = []
            data_list.append(data_info)
            
        data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list
    

@DATASETS.register_module()
class PascalContextDatasetWithScene(PascalContextDataset):
    def __init__(self, label_type='obj', label_file=None,
                 **kwargs):
        data_root = kwargs.get('data_root', None)
        self.label_type = label_type
        self.label_file = label_file
        if self.label_file and not is_abs(self.label_file) and data_root:
            self.label_file = join_path(data_root, self.label_file)
        
        super().__init__(**kwargs)
        
    def _str_to_label(self, label_str: str) -> int:
        """Convert label string to label id."""
        if label_str not in self.label_map:
            self.label_map[label_str] = len(self.label_map)
        return self.label_map[label_str]
    
    def _list_label_filter(self, label_list: List[str], filtered_val=-1) -> List[int]:
        """Filter the label list."""
        return [int(z) for z in label_list if z != str(filtered_val)]

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        
        # load label_file
        ann_info = dict()
        if not osp.isdir(self.label_file) and self.label_file:
            assert osp.isfile(self.label_file), \
                 f'Failed to load `label_file` {self.label_file}'
            lines = mmengine.list_from_file(
                 self.label_file, backend_args=self.backend_args)
            if self.label_type == 'scene':
                for line in lines:
                    img_name = line.split(' ')[0]
                    img_path = osp.join(img_dir, img_name + self.img_suffix)
                    ann_info[img_path] = self._str_to_label(line.split(' ')[1])
            elif self.label_type == 'obj':
                for line in lines:
                    img_name = line.split(' ')[0].split('.')[0]
                    img_path = osp.join(img_dir, img_name + self.img_suffix)
                    ann_info[img_path] = self._list_label_filter(
                        line.split(' ')[1:])
            else:
                raise ValueError('label_type should be either "scene" or "obj"')
            
        else:
            raise ValueError('label_file should be a file path')
        
        
        if not osp.isdir(self.ann_file) and self.ann_file:
                assert osp.isfile(self.ann_file), \
                    f'Failed to load `ann_file` {self.ann_file}'
                lines = mmengine.list_from_file(
                    self.ann_file, backend_args=self.backend_args)
                for line in lines:
                    img_name = line.strip()
                    img_path = osp.join(img_dir, img_name + self.img_suffix)
                    data_info = dict(
                        img_path=img_path)
                    if ann_dir is not None:
                        seg_map = img_name + self.seg_map_suffix
                        data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                    data_info['label_map'] = self.label_map
                    data_info['reduce_zero_label'] = self.reduce_zero_label
                    data_info['scene_label'] = ann_info[img_path]
                    data_info['seg_fields'] = []
                    data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['scene_label'] = ann_info[img_path]
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list




@TRANSFORMS.register_module()
class PackSegInputsWithScene(PackSegInputs):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
    
    def transform(self, results: dict) -> dict:
        """Method to pack the input data.
        Args:
            results (dict): Result dict from the data pipeline.
        Returns:
            dict:
            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_depth_map' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        
        if 'scene_label' in results:
            img_meta['scene_label'] = results['scene_label']
        
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        
        return packed_results