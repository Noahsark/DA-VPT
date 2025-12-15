import copy
from typing import Any, Dict, Optional, Union, Tuple
import datetime

from torch import Tensor
from mmdet.registry import MODELS
from mmengine.runner import Runner
from mmengine.runner import LogProcessor
from mmengine.registry import HOOKS, RUNNERS
from mmengine.registry import LOG_PROCESSORS
from mmengine.version import __version__
from mmengine.hooks import Hook, RuntimeInfoHook, LoggerHook
from mmengine.model import MMDistributedDataParallel
import random

from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.dist import get_dist_info
from mmengine.fileio import FileClient, get_file_backend
from mmengine.fileio import load as load_file
from mmengine.logging import print_log
from mmengine.model import BaseTTAModel, is_model_wrapper
from mmengine.runner.checkpoint import _IncompatibleKeys
from mmseg.models.segmentors import BaseSegmentor

from argparse import Namespace

from model_creator import generate_mapping

@HOOKS.register_module()
class VPTHook(Hook):
    def __init__(self, args: Namespace):
        self.args = args
    
    # def after_train_epoch(self, runner, **kwargs) -> None:
    #     model = runner.model.backbone
    #     epoch = runner.epoch
    #     # connect args
    #     self.args.proxy_prompt_len = runner.model.backbone.args.proxy_prompt_len
        
    #     if self.args.tuning_type == 'prompt' and self.args.proxy_vpt and \
    #         self.args.dynamic_kmeans > 0 and \
    #         epoch >= self.args.dynamic_kmeans:
    #         # update_mapping
    #         mapping, centroids, map_latency = generate_mapping(self.args,
    #             model.get_cls_mean_feature(), model.get_centroids()
    #         )
    #         model.update_mapping(mapping, centroids)
    #         model.reset_for_mapping_update()
        
    #     return super().after_train_epoch(runner, **kwargs)
    
    
    def after_train_iter(self, runner, batch_idx, **kwargs) -> None:
        
        if isinstance(runner.model, BaseSegmentor):
            model = runner.model.backbone
        else:
            # model is under DDP
            model = runner.model.module.backbone
        
        iter = batch_idx
        self.args.proxy_prompt_len = model.args.proxy_prompt_len
        
        if self.args.tuning_type == 'prompt' and self.args.proxy_vpt and \
            self.args.dynamic_kmeans > 0 and \
            iter >= self.args.dynamic_kmeans and \
            iter % self.args.dynamic_kmeans == 0:
            # update_mapping
            mapping, centroids, map_latency = generate_mapping(self.args,
                model.get_cls_mean_feature(), model.get_centroids()
            )
            model.update_mapping(mapping, centroids)
            model.reset_for_mapping_update()
        
        return super().after_train_iter(runner, batch_idx, **kwargs)



from mmengine.utils.dl_utils import collect_env
from collections import OrderedDict


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

@RUNNERS.register_module()
class CostumerRunner(Runner):
        
    def _log_env(self, env_cfg: dict) -> None:
        """Logging environment information of the current task.

        Args:
            env_cfg (dict): The environment config of the runner.
        """
        # Collect and log environment information.
        env = collect_env()
        runtime_env = OrderedDict()
        runtime_env.update(env_cfg)
        runtime_env.update(self._randomness_cfg)
        runtime_env['seed'] = self._seed
        runtime_env['Distributed launcher'] = self._launcher
        runtime_env['Distributed training'] = self._distributed
        runtime_env['GPU number'] = self._world_size

        env_info = '\n    ' + '\n    '.join(f'{k}: {v}'
                                            for k, v in env.items())
        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        self.logger.info('\n' + dash_line + '\nSystem environment:' +
                         env_info + '\n'
                         '\nRuntime environment:' + runtime_env_info + '\n' +
                         dash_line + '\n')
    

@HOOKS.register_module()
class LearnableTrackerHook(Hook):
    
    def before_run(self, runner) -> None:
        model = runner.model
        if isinstance(model, MMDistributedDataParallel):
            model = model.module
        
        model.print_learnable_parameters(runner.logger)


DATA_BATCH = Optional[Union[dict, tuple, list]]
@HOOKS.register_module()
class CustomerRuntimeInfoHook(RuntimeInfoHook):
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Update ``log_vars`` in model outputs every iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        log_items = runner.cfg.log_config['log_items']
        
        if outputs is not None:
            for key, value in outputs.items():
                if key in log_items:
                    runner.message_hub.update_scalar(f'train/{key}', value)



@LOG_PROCESSORS.register_module()
class CustomerLogProcessor(LogProcessor):
    
    def __init__(self, log_filter=None, **kwargs):
        super(CustomerLogProcessor, self).__init__(**kwargs)
        self.log_filter = log_filter
    
    def get_log_after_iter(self, runner, batch_idx: int,
                           mode: str) -> Tuple[dict, str]:
        """Format log string after training, validation or testing iteration.

        Args:
            runner (Runner): The runner of training phase.
            batch_idx (int): The index of the current batch in the current
                loop.
            mode (str): Current mode of runner, train, test or val.

        Return:
            Tuple[dict, str]: Formatted log dict/string which will be
            recorded by :obj:`runner.message_hub` and :obj:`runner.visualizer`.
        """
        assert mode in ['train', 'test', 'val']
        # Overwrite ``window_size`` defined in ``custom_cfg`` to int value.
        parsed_cfg = self._parse_windows_size(runner, batch_idx,
                                              self.custom_cfg)
        # log_tag is used to write log information to terminal
        log_tag = self._collect_scalars(parsed_cfg, runner, mode)

        if self.log_filter is not None:
            log_tag = {k: v for k, v in log_tag.items() if not k in self.log_filter}

        # If `self.log_with_hierarchy` is False, the tag is the same as
        # log_tag. Otherwise, each key in tag starts with prefix `train`,
        # `test` or `val`
        if not self.log_with_hierarchy:
            tag = copy.deepcopy(log_tag)
        else:
            tag = self._collect_scalars(parsed_cfg, runner, mode, True)

        # Record learning rate.
        lr_str_list = []
        for key, value in tag.items():
            if key.endswith('lr'):
                key = self._remove_prefix(key, f'{mode}/')
                log_tag.pop(key)
                lr_str_list.append(f'{key}: '
                                   f'{value:.{self.num_digits}e}')
        lr_str = ' '.join(lr_str_list)
        # Format log header.
        # by_epoch == True
        #   train/val: Epoch [5][5/10]  ...
        #   test: Epoch [5/10]
        # by_epoch == False
        #  train: Epoch [5/10000] ... (divided by `max_iter`)
        #  val/test: Epoch [5/2000] ... (divided by length of dataloader)
        
        if self.by_epoch:
            # Align the iteration log:
            # Epoch(train)  [  9][010/270]
            # ...                 ||| |||
            # Epoch(train)  [ 10][100/270]
            dataloader_len = self._get_dataloader_size(runner, mode)
            cur_iter = self._get_iter(runner, batch_idx)
            cur_iter_str = str(cur_iter).rjust(len(str(dataloader_len)))
            if mode in ['train', 'val']:
                cur_epoch = self._get_epoch(runner, mode)
                if not (isinstance(runner._train_loop, dict)
                        or runner._train_loop is None):
                    # Right Align the epoch log:
                    # Epoch(train)   [9][100/270]
                    # ...             ||
                    # Epoch(train) [100][100/270]
                    max_epochs = runner.max_epochs
                    # 3 means the three characters: "[", "]", and " " occupied
                    # in " [{max_epochs}]"
                    cur_epoch_str = f'[{cur_epoch}]'.rjust(
                        len(str(max_epochs)) + 3, ' ')
                else:
                    cur_epoch_str = f'[{cur_epoch}]'
                tag['epoch'] = cur_epoch
                log_str = (f'Epoch({mode}){cur_epoch_str}'
                           f'[{cur_iter_str}/{dataloader_len}] ')
            else:
                log_str = (f'Epoch({mode}) '
                           f'[{cur_iter_str}/{dataloader_len}] ')
        else:
            if mode == 'train':
                cur_iter = self._get_iter(runner, batch_idx)
                cur_iter_str = str(cur_iter).rjust(len(str(runner.max_iters)))
                log_str = (f'Iter({mode}) '
                           f'[{cur_iter_str}/{runner.max_iters}]  ')
            else:
                dataloader_len = self._get_dataloader_size(runner, mode)
                cur_iter_str = str(batch_idx + 1).rjust(
                    len(str(dataloader_len)))
                log_str = (f'Iter({mode}) [{cur_iter_str}/{dataloader_len}]  ')
        
        # Add global iter.
        if isinstance(runner._train_loop, dict) or runner._train_loop is None:
            tag['iter'] = 0
        else:
            tag['iter'] = runner.iter + 1
        # Concatenate lr, momentum string with log header.
        log_str += f'{lr_str} '
        
        # If IterTimerHook used in runner, eta, time, and data_time should be
        # recorded.
        if (all(item in log_tag for item in ['time', 'data_time'])
                and 'eta' in runner.message_hub.runtime_info):
            eta = runner.message_hub.get_info('eta')
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            log_str += colorstr('yellow', f'eta: {eta_str} ')
            
            # log_str += (f'time: {log_tag["time"]:.{self.num_digits}f}  '
            #             f'data_time: '
            #             f'{log_tag["data_time"]:.{self.num_digits}f}  ')
            
            # Pop recorded keys
            log_tag.pop('time')
            log_tag.pop('data_time')

        # If cuda/musa is available,
        # the max memory occupied should be calculated.
        # if is_cuda_available() or is_musa_available():
        #     max_memory = self._get_max_memory(runner)
        #     log_str += f'memory: {max_memory}  '
        #     tag['memory'] = max_memory

        # Loop left keys to fill `log_str`.
        if mode in ('train', 'val'):
            log_items = []
            for name, val in log_tag.items():
                if mode == 'val' and not name.startswith('val/loss'):
                    continue
                if isinstance(val, float):
                    val = f'{val:.{self.num_digits}f}'
                log_items.append(f'{name}: {val}')
            log_str += ' '.join(log_items)
        return tag, log_str