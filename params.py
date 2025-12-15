import argparse
from pathlib import Path

def model_parameters(parser):
    parser.add_argument("--model_name", type=str, default="vit-B",
                        help="architecture")
    parser.add_argument("--drop_path", type=float, default=0.0, help='Drop path rate')
    parser.add_argument("--fp16", action="store_true", help="use mix precision")
    parser.add_argument("--local_model_path", type=str, default="./save/linear-vit-b-300ep.pth.tar",
                        help="local model") # For models that are not ready in timm, (e.g. MOCO v3)
    
    
    return parser
    
def prompt_parameters(parser):
    parser.add_argument('--num_prompts', type=int, default=20, 
                        help="number of prompt tokens.")
    parser.add_argument("--proxy_prompt_len", type=int, default=10, 
                        help="number of guidance prompt tokens.")
    # guided idx = [proxy_prompt_start_idx, proxy_prompt_start_idx + proxy_prompt_depth)
    parser.add_argument("--proxy_prompt_start_idx", type=int, default=11, 
                        help="where the guide start") #NOTE: inclusive
    parser.add_argument("--proxy_prompt_end_idx", type=int, default=11) #NOTE: inclusive
    parser.add_argument("--prompt_drop_rate", type=float, default=0.0)
    return parser

def criterion_parameters(parser):
    # Metric Learning Parameters
    parser.add_argument("--proxy_vpt", action="store_true", help="Use proxy anchor")
    parser.add_argument("--criterion", type=str, default="proxyanchor", choices=["proxyanchor", "proxynca"])
    parser.add_argument("--loss_oproxy_pos_alpha", type=float, default=32.0, help="positive alpha")
    parser.add_argument("--loss_oproxy_neg_alpha", type=float, default=32.0, help="negative alpha")
    parser.add_argument("--loss_oproxy_pos_delta", type=float, default=0.1, help="positive delta")
    parser.add_argument("--loss_oproxy_neg_delta", type=float, default=-0.1, help="negative delta")
    
    parser.add_argument("--head_pooling", type=str, default="max", choices=["mean", "max"])
    parser.add_argument("--local_layer_gradient", action="store_true", help="pass gradient to vpt")
    
    # VPT Mapping Parameters
    parser.add_argument("--initial_mapping", type=str, default="kmeans", 
                        choices=["random", "kmeans", "kmeans++", "all_classes", 
                                 "balanced", "inverse_mapping", "inverse_mapping++"])
    parser.add_argument("--init_sample_per_class", type=int, default=10, help="samples per class")
    parser.add_argument("--select_type", type=str, default="cls_attn", 
                        choices=["cross_attn", "cls_attn"])
    parser.add_argument("--pooling_type", type=str, default="mean", choices=["mean", "max"])
    
    parser.add_argument("--kmeans_norm", type=str, default="layer", choices=["none", "l2", "layer"])
    parser.add_argument("--vpt_norm", type=str, default="none", choices=["none", "l2", "layer"])
    parser.add_argument("--init_divide_by8", action="store_true", help="init divide by 8")
    
    parser.add_argument("--vpt_loss_weight", type=float, default=0.0, help="proxy ratio")
    parser.add_argument("--cls_loss_weight", type=float, default=0.0, help="cls loss ratio")
    
    # Dynamic Bias
    parser.add_argument("--learn_bias", action="store_true", help="learn bias")
    parser.add_argument("--train_cls", action="store_true", help="train cls token")
    parser.add_argument("--bias_fc1", action="store_true", help="bias fc1")
    parser.add_argument("--bias_fc2", action="store_true", help="bias fc2")
    parser.add_argument("--bias_proj", action="store_true", help="bias attn_proj")
    parser.add_argument("--bias_norm1", action="store_true", help="norm norm1")
    parser.add_argument("--bias_norm2", action="store_true", help="norm norm2")
    parser.add_argument("--bias_norm", action="store_true", help="norm norm")
    parser.add_argument("--norm_norm1", action="store_true", help="norm norm1")
    parser.add_argument("--norm_norm2", action="store_true", help="norm norm2")
    parser.add_argument("--norm_norm", action="store_true", help="norm norm")
    parser.add_argument("--bias_q", action="store_true", help="bias q")
    parser.add_argument("--bias_k", action="store_true", help="bias k")
    parser.add_argument("--bias_v", action="store_true", help="bias v")

    parser.add_argument("--vpt_head_init", action="store_true", help="init head")
    parser.add_argument("--vpt_init", action="store_true", help="init pooling")
    parser.add_argument("--init_pooling_type", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--init_select_topk", type=int, default=5)
    parser.add_argument("--noise_ratio", type=float, default=0.2)
    parser.add_argument("--dynamic_kmeans", type=int, default=0, 
                    help="which epoch start to do the dynamic kmeans, 0 means never")
    parser.add_argument("--vpt_cls_loss_weight", type=float, default=0.1, help="cls loss ratio")
    parser.add_argument("--vpt_cls_loss", action="store_true", help="use head")
    parser.add_argument("--connection_type", type=str, default="none", choices=["none", 
            "single_gate", "linear_gate_1", "linear_gate_2", "linear_gate_3", "fixed"])
    parser.add_argument("--fixed_gate_ratio", type=float, default=0.2)
    parser.add_argument("--connect_last_layer", action="store_true", help="connect last layer")
    parser.add_argument("--attention_scaling_topk", action="store_true", help="use head")
    parser.add_argument("--loss_vpt_scale_ratio", type=float, default=1.0,
                        help="Scale vpt in loss before cal sims")
    parser.add_argument("--mae_pooling", type=str, default="all", choices=["img", "all"])
    parser.add_argument("--patch_type", type=str, default="output", choices=["patch", "query", "key", "output"])
    parser.add_argument("--vpt_type_for_cls_ml", type=str, default="vpt", choices=["vpt", "query", "key"])
    parser.add_argument("--vpt_type_for_patch_ml", type=str, default="vpt", choices=["vpt", "query", "key"])
    parser.add_argument("--cls_type", type=str, default="query", choices=["cls", "query", "key"])
    parser.add_argument("--mask", action="store_true", help="prompt mask")
    
    return parser


def dataset_parameters(parser):
    parser.add_argument("--dataset", type=str, default='cub')
    parser.add_argument("--data_dir", type=str, default="./vpt_data", help="dataset path")
    parser.add_argument("--class_sampler", action="store_true", help="Use class sampler")
    parser.add_argument("--samples_per_class", type=int, default=2, help="samples per class")
    parser.add_argument("--data_cropsize", type=int, default=224, help="crop size")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    return parser

def training_parameters(parser):
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--wd_head", type=float, default=3.0)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR', 
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N', help='epochs to warmup LR')
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--eval_after", type=int, default=60, help="evaluate after n epochs")
    parser.add_argument('--warmup_lr', type=float, default=1e-7, metavar='LR',
                    help='warmup learning rate (default: 1e-7)')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
    parser.add_argument('--sched_on_updates', action='store_true', 
                        help='Schedule on number of updates instead of epochs')
    parser.add_argument("--early_stop_thr", type=float, default=-1, help="early stop threshold")
    parser.add_argument("--vpt_loss_stop_thr", type=float, default=-1, help="vpt loss threshold")
    parser.add_argument("--open_bias_decay", action="store_true", help="open bias decay")
    parser.add_argument("--open_weight_decay", action="store_true", help="open weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", 
                        choices=["adam", "adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true", help="nesterov")
    parser.add_argument("--save_checkpoint", action="store_true", help="save best checkpoint")
    parser.add_argument("--save_dir", type=str, default="../save/checkpoint", help="save path")
    parser.add_argument("--task_name", type=str, default="test", help="task name")
    parser.add_argument("--load_model", type=str, help="ckpt's path to load model")
    parser.add_argument("--load_config", type=str, default="", help="config's path to load model")
    parser.add_argument("--test_vpt_loss_off", action="store_true", help="vpt loss off")
    parser.add_argument("--test_cls_loss_off", action="store_true", help="cls loss off")
    
    return parser

# tuning parameters
def tuning_parameters(parser):
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--tuning_type", type=str, default="prompt", choices=["full", "linear", "prompt"])
    parser.add_argument("--quiet_mode", action="store_true", help="quite mode")
    return parser




_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off"}

def _coerce(value: str, current):
    v = value.strip().strip('"').strip("'")

    # Booleans
    if isinstance(current, bool):
        lv = v.lower()
        if lv in _TRUE: return True
        if lv in _FALSE: return False
        return current  # ignore bad cast

    # Int / float
    if isinstance(current, int) and not isinstance(current, bool):
        try: return int(v, 0)   # supports 10, 0x10
        except ValueError: return current
    if isinstance(current, float):
        try: return float(v)
        except ValueError: return current

    # Simple lists/tuples via comma split
    if isinstance(current, (list, tuple)):
        parts = [p.strip() for p in v.split(",")] if v else []
        if current:  # cast to the type of the first element if available
            t = type(current[0])
            try:
                parts = [t(p) for p in parts]
            except Exception:
                pass
        return parts if isinstance(current, list) else tuple(parts)

    # Path-like
    if isinstance(current, Path):
        return Path(v)

    # None or everything else: try constructing with current's type, else keep string
    try:
        return type(current)(v) if current is not None else v
    except Exception:
        return v


def update_args_from_file_colon(filename: str, args: argparse.Namespace) -> argparse.Namespace:
    with open(filename, "r", encoding="utf-8") as f:
        print(f"reading config from {filename}")
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue  # ignore malformed
            key, val = line.split(":", 1)
            key = key.strip().lstrip("-").replace("-", "_")
            if not hasattr(args, key):
                continue
            cur = getattr(args, key)
            new_val = _coerce(val, cur)
            setattr(args, key, new_val)
    return args



def parse_args():
    parser = argparse.ArgumentParser()
    parser = model_parameters(parser)
    parser = prompt_parameters(parser)
    parser = dataset_parameters(parser)
    parser = training_parameters(parser)
    parser = tuning_parameters(parser)
    parser = criterion_parameters(parser)
    parser.add_argument("--config", type=str, default="configs/fgvc/vitb_cub200.yaml")

    args = parser.parse_args()
    args = update_args_from_file_colon(args.config, args)

    return args