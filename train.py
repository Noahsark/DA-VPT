"""
DA-VPT Training Script - Cleaned and Improved Version
Distribution Aware Visual Prompt Tuning for Vision Transformers
"""

import os
import socket
import time
import yaml
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from timm.scheduler.scheduler_factory import create_scheduler

from models.vpt import *
from Dataset.FGVC_json import *
from Dataset.VTAB_txt import *
from Dataset.class_sampler import *
from Dataset.torch_vision import *
from Dataset.names import _FGVC_CATALOG, _VTAB_CATALOG, _TORCH_VISION_CATALOG
from utils.utils import *
from params import parse_args
from model_creator import create_model, generate_mapping


_VERSION = "DEMO"


def get_img_dict(dataset):
    """Create class indices dictionary for torch vision datasets."""
    class_indices = {i: [] for i in range(len(dataset.classes))}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    return class_indices


def create_dataset(args):
    """Create train and validation datasets based on dataset type."""
    if args.dataset in _FGVC_CATALOG:
        train_dataset, _, val_dataset, num_class = create_fgvc_dataset(args)
    elif args.dataset in _VTAB_CATALOG:
        train_dataset, val_dataset, num_class = create_vtab_dataset(args)
    elif args.dataset in _TORCH_VISION_CATALOG:
        train_dataset, val_dataset, num_class = create_tv_dataset(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")
    
    # Create train loader
    if args.class_sampler:
        if args.dataset not in _TORCH_VISION_CATALOG:
            train_sampler = ClassSampler(
                args, train_dataset.get_img_dict(), train_dataset.get_num_imgs()
            )
        else:
            train_sampler = ClassSampler(
                args, get_img_dict(train_dataset), len(train_dataset), has_path=False
            )
        train_loader = DataLoader(
            train_dataset,
            num_workers=args.workers,
            pin_memory=True,
            batch_sampler=train_sampler
        )
    else:
        train_sampler = RandomSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )
    
    # Create validation loader
    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False
    )
    
    return train_loader, val_loader, num_class


def train_one_epoch(args, model, train_loader, optimizer, criterion, 
                   epoch, scaler, context, lr_scheduler=None):
    """Train model for one epoch."""
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    vpt_loss = AverageMeter()
    latency = AverageMeter()
    
    # Setup progress bar
    if not args.quiet_mode:
        data_iterator = tqdm(
            train_loader,
            bar_format='{desc}{percentage:2.2f}% [{n_fmt}/{total_fmt},'
                      '{elapsed}{postfix}]',
            ncols=96, ascii=True, 
            desc=f'[GPU:{args.gpu} Ep:{epoch}]: '
        )
    else:
        data_iterator = train_loader
    
    # Set model to training mode
    if args.tuning_type == 'prompt':
        model.reset_for_mapping_update()
    model.train()
    
    # Check if dynamic mapping update is needed
    update_mapping = (
        args.tuning_type == 'prompt' and 
        args.dynamic_kmeans > 0 and 
        epoch >= args.dynamic_kmeans
    )
    
    step_per_epoch = len(train_loader)
    num_updates = epoch * step_per_epoch
    
    for step, (images, labels) in enumerate(data_iterator):
        torch.cuda.synchronize()
        start_time = time.time()

        # Adjust learning rate if no scheduler
        if lr_scheduler is None:
            adjust_learning_rate(optimizer, step / step_per_epoch + epoch, args)

        # Move data to GPU
        images = images.cuda(0, non_blocking=True)
        labels = labels.cuda(0, non_blocking=True)
        
        vpt_loss_val = torch.tensor(0.0, device=images.device)
        
        # Forward pass
        with context:
            if args.tuning_type == 'prompt':
                logits, vpt_loss_val, vpt_logits = model(
                    images, labels=labels, update_mapping=update_mapping
                )
            else:
                logits = model(images)
            
            classification_loss = criterion(logits, labels)   
            total_loss = classification_loss + vpt_loss_val
        
        # Calculate accuracy and update metrics
        acc1, _ = accuracy(logits, labels, topk=(1, 1))
        train_loss.update(classification_loss.item(), images.size(0))
        train_acc.update(acc1[0].item(), images.size(0))
        vpt_loss.update(vpt_loss_val.item(), images.size(0))

        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
        scaler.step(optimizer)     
        scaler.update()
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        num_updates += 1
        
        # Update learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=train_loss.avg)
        
        latency.update(time.time() - start_time)
        
        # Update progress bar
        if not args.quiet_mode:
            postfix = '{:.1f}ms l: {:.2f} vptl: {:.2f} lr: {:.1e}'.format(
                1000 * latency.avg, train_loss.val, vpt_loss.val, 
                optimizer.param_groups[0]['lr']
            )
            data_iterator.set_postfix_str(postfix)
    
    # Update mapping if needed
    map_latency = 0
    if (update_mapping and args.initial_mapping != 'all_classes' 
        and args.proxy_prompt_len > 0):
        mapping, centroids, map_latency = generate_mapping(
            args, model.get_cls_mean_feature(), model.get_centroids()
        )
        model.update_mapping(mapping, centroids)
        model.reset_for_mapping_update()
    
    # Print summary for quiet mode
    if args.quiet_mode:
        summary = (
            f'[GPU:{args.gpu} Ep:{epoch}]: t:{latency.sum:.1f}s '
            f'l: {1000 * latency.avg:.1f}ms loss: {train_loss.val:.2f} '
            f'vloss: {vpt_loss.val:.2f} lr: {optimizer.param_groups[0]["lr"]:.1e}'
        )
        print(colorstr('yellow', summary))
    
    return train_loss.avg, train_acc.avg, vpt_loss.avg, map_latency


def validate(model, val_loader, criterion, args, quiet_mode=True):
    """Validate model on validation set."""
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    if not quiet_mode:
        data_iterator = tqdm(
            val_loader,
            bar_format='{desc}{percentage:2.2f}% [{n_fmt}/{total_fmt},'
                      '{elapsed}{postfix}]',
            ncols=96, ascii=True, 
            desc=f'[GPU:{args.gpu} Val]: '
        )
    else:
        data_iterator = val_loader
    
    model.eval()
    with torch.no_grad():
        for step, (images, labels) in enumerate(data_iterator):
            images = images.cuda(0, non_blocking=True)
            labels = labels.cuda(0, non_blocking=True)

            logits, _, _ = model(images)
            loss = criterion(logits, labels)

            acc1, _ = accuracy(logits, labels, topk=(1, 1))
            val_loss.update(loss.item(), images.size(0))
            val_acc.update(acc1[0].item(), images.size(0))
            
    return val_loss.avg, val_acc.avg


# def load_config(args):
#     """Load configuration from file if offline mode."""
        
#     if not args.load_config or not os.path.exists(args.load_config):
#         print("No config file, using input settings.")
#         return
        
#     with open(args.load_config) as file:
#         config = yaml.load(file, Loader=yaml.FullLoader)
#         if config is None:
#             raise FileNotFoundError("Load config file not found.")
        
#         for key, value in config.items():
#             setattr(args, key, value['value'])


def setup_environment(args):
    """Setup CUDA environment and device."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["WORLD_SIZE"] = "1"
    
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    set_seed(args.seed)
    
    return device


def create_optimizer(args, param_groups):
    """Create optimizer based on args."""
    optimizer_map = {
        'adamw': torch.optim.AdamW,
        'sgd': lambda params, lr: torch.optim.SGD(
            params, lr=lr, momentum=args.momentum, nesterov=args.nesterov
        ),
        'adam': torch.optim.Adam
    }
    
    if args.optimizer not in optimizer_map:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")
    
    return optimizer_map[args.optimizer](param_groups, lr=args.lr)


def save_checkpoint(args, model, epoch, best_acc, is_best):
    """Save model checkpoint if needed."""
    if not args.save_checkpoint or not is_best:
        return
        
    save_model(args, model, epoch, best_acc)


def should_early_stop(args, epoch, val_acc, best_acc, vpt_epoch_loss):
    """Check if training should stop early."""
    # VPT loss early stopping
    if args.vpt_loss_stop_thr > 0 and vpt_epoch_loss < args.vpt_loss_stop_thr:
        print(colorstr('red', f'Early stop at epoch: {epoch} '
                             f'for vpt_epoch_loss: {vpt_epoch_loss:.4f}'))
        return True
    
    # Validation accuracy early stopping
    if (val_acc < best_acc and args.early_stop_thr > 0 
        and val_acc < args.early_stop_thr):
        print(colorstr('red', f'Early stop at epoch: {epoch}'))
        return True
    
    return False


def train():
    """Main training function."""
    args = parse_args()
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    
    # Setup logging and configuration
    # load_config(args)
    device = setup_environment(args)
    
    # Print settings
    print(colorstr('green', "\n" + "="*60))
    print(colorstr('green', "Settings"))
    print(colorstr('green', "="*60))
    print(f'Server: {socket.gethostname()}')
    print(f'Experiment Version: {_VERSION}')
    print(f'Model: {args.model_name}')
    
    # Setup mixed precision context
    context = autocast() if args.fp16 else nullcontext()
    
    # Create datasets
    train_loader, val_loader, num_classes = create_dataset(args)
    args.num_classes = num_classes
    
    # Adjust proxy prompt length if needed
    if args.num_classes < args.proxy_prompt_len:
        args.proxy_prompt_len = args.num_classes
        args.initial_mapping = 'all_classes'
    
    # Create model
    model = create_model(
        args=args, num_class=num_classes, context=context, device=device
    )
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Setup optimizer
    param_groups = param_groups_weight_decay(
        args, model=model, 
        weight_decay=args.weight_decay, 
        weight_decay_head=args.wd_head
    )
    optimizer = create_optimizer(args, param_groups)
    
    # Setup learning rate scheduler
    updates_per_epoch = len(train_loader)
    lr_scheduler, _ = create_scheduler(
        args, optimizer, updates_per_epoch=updates_per_epoch
    )
    
    # Setup loss and scaler
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()
    
    best_acc = 0
    print(colorstr('green', "\n" + "="*60))
    print(colorstr('green', "Start Training"))
    print(colorstr('green', "="*60))
    
    # Training loop
    for epoch in range(args.epochs):
        train_epoch_loss, train_acc, vpt_epoch_loss, map_latency = train_one_epoch(
            args=args, model=model, train_loader=train_loader,
            optimizer=optimizer, criterion=criterion, epoch=epoch,
            scaler=scaler, context=context, lr_scheduler=lr_scheduler
        )
        
        # Update learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1, train_acc)
        
        # Check for early stopping
        if should_early_stop(args, epoch, -1, best_acc, vpt_epoch_loss):
            break
        
        # Validation
        if epoch >= args.eval_after:
            val_epoch_loss, val_acc = validate(
                model=model, val_loader=val_loader, 
                criterion=criterion, args=args
            )
            
            summary = (f"TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, "
                      f"lr: {optimizer.param_groups[0]['lr']:.1e}")
            print(colorstr('green', summary))
            
            # Save best model
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            save_checkpoint(args, model, epoch, best_acc, is_best)
            
            # Check for early stopping
            if should_early_stop(args, epoch, val_acc, best_acc, vpt_epoch_loss):
                break
            
            state = {
                'best_acc': best_acc,
                'train_acc': train_acc,
                'train_loss': train_epoch_loss,
                'vpt_loss': vpt_epoch_loss,
                'val_acc': val_acc,
                'val_loss': val_epoch_loss
            }
        else:
            state = {
                'train_acc': train_acc,
                'train_loss': train_epoch_loss,
                'vpt_loss': vpt_epoch_loss,
                'map_latency': map_latency
            }

    # Cleanup
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    train()