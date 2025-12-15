from collections import OrderedDict, Counter, deque
import torch, timm
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from contextlib import nullcontext
import yaml, termcolor, time, math
from scipy.optimize import linprog

#################### costumer imports ###################
from models.vpt import *
from models.vit import *
from Dataset.FGVC_json import *
from Dataset.VTAB_txt import *
from Dataset.class_sampler import *
from Dataset.mapping_sampler import *
from Dataset.torch_vision import *
from Dataset.names import _FGVC_CATALOG, _VTAB_CATALOG, _TORCH_VISION_CATALOG, _MODEL_CATALOG
from utils.utils import *
from params import *
import time
#####################################################


###########################################################################

def load_model_parameters(source_model, target_model):
    # load parameters from source_model to target_model
    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()
    #WARNING filter out head, because shape mismatch
    state_dict = {k:v for k, v in source_dict.items() if k in target_dict and 'head' not in k}
    target_dict.update(state_dict)
    target_model.load_state_dict(target_dict)
    
    
def filter_param_names(ckpt, name="module"):
    for k in list(ckpt.keys()):
        if k.startswith(name):
            new_key = k.replace(name + '.', "")
            ckpt[new_key] = ckpt.pop(k)
    return ckpt


def create_model(args, num_class, **kwargs):
    
    _SEMANTIC_MAPPING = ["kmeans", "kmeans++"]
    
    model = None

    if not args.model_name in _MODEL_CATALOG and not args.model_name.split('-')[0] == "moco":
        raise ValueError("Model {} not supported.".format(args.model_name))
    
    if args.tuning_type in ["full", "linear"]:
        
        if not args.model_name.split('-')[0] == "moco":
            model = timm.create_model(_MODEL_CATALOG[args.model_name], pretrained=True)
        else:
            model = create_vit(args=args, num_classes=num_class)
            #note: load moco model
            ckpt = torch.load(args.local_model_path, map_location='cpu')['state_dict']
            ckpt = filter_param_names(ckpt)
            model.load_state_dict(ckpt, strict=False)
            model.separate_qkv_params()
            for name, param in model.named_parameters():
                param.requires_grad = True
            
            
        if args.tuning_type == "linear":
            for name, param in model.named_parameters():
                names = name.split('.')
                if names[0] in ['head']:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                
                
    elif args.tuning_type in ["prompt"]:
        if not args.model_name.split('-')[0] == "moco":
            pretrain_model = timm.create_model(_MODEL_CATALOG[args.model_name], pretrained=True)
        else:
            pretrain_model = create_vit(args=args, num_classes=num_class)
            #note: load moco model
            ckpt = torch.load(args.local_model_path, map_location='cpu')['state_dict']
            ckpt = filter_param_names(ckpt)
            pretrain_model.load_state_dict(ckpt, strict=False)
            
        pretrain_model.to(kwargs.get('device', torch.device('cuda')))
        context = kwargs.get('context', nullcontext())
        
        
        ##################################################################
        prompt_init = None
        last_prompt_init = None
        centroids = None
        mapping = None
        if args.initial_mapping in _SEMANTIC_MAPPING and args.proxy_prompt_len > 0:
            
            mapping, centroids, prompt_init, last_prompt_init = \
                generate_semantic_mapping(args, pretrain_model, context=context)
        
        elif args.initial_mapping == 'all_classes' or args.proxy_prompt_len == 0:
            
            mapping = torch.Tensor(range(num_class))
            
        elif args.initial_mapping == 'balanced':
            mapping = static_random_balanced_mapping(num_class, args.proxy_prompt_len)
        else:
            raise ValueError("Invalid initial mapping method.")

        #####################################################
        global_pool = args.model_name.split('-')[0] == "mae"
        vit_size = args.model_name.split('-')[1]
        assert vit_size in ["S", "B", "L", "H"]
        model = create_vpt_vit(args=args, 
                               vit_size=vit_size,
                               prompt_init=prompt_init, 
                               last_prompt_init=last_prompt_init,
                               drop_path_rate=args.drop_path, 
                               mapping=mapping, 
                               num_classes=num_class,
                               global_pool=global_pool)
        load_model_parameters(pretrain_model, model)
        model.reset_for_mapping_update()
        model.update_mapping(mapping, centroids)
        
        del pretrain_model
        
        model.separate_qkv_params()
        
        learnable_list = ['prompt_token', "head"]
        
        learnable_list.append("proxy_prompt_tokens")
        learnable_list.append("other_prompt_tokens")
        learnable_list.append("vpt_norm")
        
        if args.learn_bias:
            learnable_list.append("bias")
        if args.bias_q:
            learnable_list.append("q.bias")
        if args.bias_k:
            learnable_list.append("k.bias")
        if args.bias_v:
            learnable_list.append("v.bias")
        if args.bias_fc1:
            learnable_list.append("fc1.bias")
        if args.bias_fc2:
            learnable_list.append("fc2.bias")
        if args.bias_proj:
            learnable_list.append("proj.bias")
        if args.bias_norm1:
            learnable_list.append("norm1.bias")
        if args.bias_norm2:
            learnable_list.append("norm2.bias")
        if args.bias_norm:
            learnable_list.append("norm.bias")
        if args.norm_norm1:
            learnable_list.append("norm1")
        if args.norm_norm2:
            learnable_list.append("norm2")
        if args.norm_norm:
            learnable_list.append("norm")
        if args.train_cls:
            learnable_list.append("cls_token")
        
        model.set_learnable_parameters(learnable_list)
        #note: ##########################################################################

    # modify the output dimension (num classes) according to the task
    in_features = model.head.in_features
    model.head = torch.nn.Linear(in_features=in_features, out_features=num_class)
    trunc_normal_(model.head.weight, std=.02)
    nn.init.constant_(model.head.bias, 0)
    
    if args.vpt_cls_loss:
        model.vpt_head = nn.Linear(in_features=in_features, out_features=num_class)
        trunc_normal_(model.vpt_head.weight, std=.02)
        nn.init.constant_(model.vpt_head.bias, 0)
    if args.tuning_type == "prompt":
        model.print_learnable_parameters()
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name + ': ' + str(list(param.shape)))
        # print num of parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of params (M): %.4f' % (num_params / 1.e6))
        
    return model


#Note: fixed the bug when len(centroids) < k
@torch.no_grad()
def cosine_kmeans(data, k, centroids=None, tol=1e-4, max_iter=20, norm='none'):
    # data: num_classes x 768
    # k: number of prompts
    # centroids: k x 768

    # Identify zero classes
    zero_classes = (data == 0).all(dim=1)
    non_zero_data = data[~zero_classes]

    if norm == "l2":
        non_zero_data = F.normalize(non_zero_data, p=2, dim=1)
    elif norm == "layer":
        # Warning: no trainable weight/bias applied
        non_zero_data = F.layer_norm(non_zero_data, non_zero_data.shape[1:])
    else:
        pass

    if centroids is None:
        # Ensure that the number of centroids does not exceed the number of non-zero data points
        if non_zero_data.size(0) < k:
            raise ValueError(f"Number of non-zero data points ({non_zero_data.size(0)}) is less than k ({k}).")
        indices = torch.randperm(non_zero_data.size(0))[:k]
        centroids = non_zero_data[indices]

    for _ in range(max_iter):
        # Compute cosine similarity between non-zero data points and centroids
        similarities = F.cosine_similarity(non_zero_data[:, None], centroids[None, :], dim=2)
        
        # Assign clusters based on the highest cosine similarity
        cluster_assignments = similarities.argmax(dim=1)
        
        # Update centroids
        new_centroids = []
        for i in range(k):
            assigned_data = non_zero_data[cluster_assignments == i]
            if assigned_data.size(0) > 0:
                new_centroid = assigned_data.mean(dim=0)
            else:
                # Keep the old centroid if no data points are assigned
                new_centroid = centroids[i]
                # Alternatively, reinitialize the centroid:
                # new_centroid = non_zero_data[torch.randint(0, non_zero_data.size(0), (1,))]
            new_centroids.append(new_centroid)
        new_centroids = torch.stack(new_centroids)
        
        # Check for convergence
        if torch.norm(centroids - new_centroids, dim=1).sum() < tol:
            break
        
        centroids = new_centroids

    # Randomly assign zero classes to centroids
    zero_class_assignments = torch.randint(k, (zero_classes.sum(),), device=data.device)
    
    # Combine cluster assignments for non-zero and zero classes
    combined_assignments = torch.empty(data.size(0), dtype=torch.long, device=data.device)
    combined_assignments[~zero_classes] = cluster_assignments
    combined_assignments[zero_classes] = zero_class_assignments
    
    return centroids, combined_assignments



@torch.no_grad()
def cosine_kmeans_origin(data, k, centroids=None, tol=1e-4, max_iter=20, norm='none'):
    # data: num_classes x 768
    # k: number of prompts
    # centroids: k x 768
    
    # Randomly initialize centroids by selecting k unique data points
    assert data.size(0) >= k
    #note: 1.2.2.1
    if norm == "l2":
        data = F.normalize(data, p=2, dim=1)
    elif norm == "layer":
        # warning: no trainable weight/bias applied
        data = F.layer_norm(data, data.shape[1:])
    else:
         pass
    
    if centroids is None:
        indices = torch.randperm(data.size(0))[:k]
        centroids = data[indices] # k x D
    
    for _ in range(max_iter):
        # Compute cosine similarity between data points and centroids
        # data: N x 1 x D, centroids: 1 x k x D -> similarities: N x k
        similarities = F.cosine_similarity(data[:, None], centroids[None, :], dim=2)

        # Assign clusters based on the highest cosine similarity
        cluster_assignments = similarities.argmax(dim=1) # N x k -> N

        # Update centroids
        new_centroids = torch.stack([
            data[cluster_assignments == i].mean(dim=0) 
            for i in range(k)
        ])

        # Check for convergence
        if torch.norm(centroids - new_centroids, dim=1).sum() < tol:
            break
        centroids = new_centroids
    
    return centroids, cluster_assignments


# kmeans++
@torch.no_grad()
def cosine_kmeans_plus_equal_size(data, k, centroids=None, tol=1e-4, max_iter=100, 
                        balance_iter=50, norm='none'):
    # data: num_classes x 768
    # k: number of prompts
    # centroids: k x 768
    
    assert data.size(0) >= k
    if norm == "l2":
        data = F.normalize(data, p=2, dim=1)
    elif norm == "layer":
        data = F.layer_norm(data, data.shape[1:])
    else:
         pass
    
    # k-means++ initialization
    if centroids is None:
        centroids = torch.empty(k, data.size(1)).to(data.device)

    # Randomly select the first centroid
    index = torch.randint(0, data.size(0), (1,))
    centroids[0] = data[index]

    # Select the rest centroids
    for i in range(1, k):
        distances = torch.min(torch.cdist(data, centroids[:i], p=2) ** 2, dim=1)[0]
        probabilities = distances / distances.sum()
        index = torch.multinomial(probabilities, 1)
        centroids[i] = data[index]

    # Main k-means loop
    for _ in range(max_iter):
        # Compute cosine similarity
        similarities = F.cosine_similarity(data[:, None], centroids[None, :], dim=2)
        cluster_assignments = similarities.argmax(dim=1)

        # Update centroids with a view to balance sizes
        for _ in range(balance_iter):
            sizes = torch.bincount(cluster_assignments, minlength=k)
            max_size = (data.size(0) // k) + 1
            min_size = data.size(0) // k

            for idx in range(k):
                if sizes[idx] > max_size:
                    # Points to potentially reassign
                    points_idx = torch.where(cluster_assignments == idx)[0]
                    excess = sizes[idx] - max_size
                    points_to_reassign = points_idx[:excess]

                    # Compute similarity for points to potentially reassign
                    sim = similarities[points_to_reassign]
                    sim[:, idx] = -1e10  # Make current cluster non-eligible
                    new_clusters = sim.argmax(dim=1)

                    cluster_assignments[points_to_reassign] = new_clusters

        new_centroids = torch.stack([
            data[cluster_assignments == i].mean(dim=0) 
            for i in range(k)
        ])

        # Check for convergence
        if torch.norm(centroids - new_centroids, dim=1).sum() < tol:
            break
        centroids = new_centroids

    return centroids, cluster_assignments

@torch.no_grad()
def balanced_cosine_kmeans(data, k, tol=1e-4, max_iter=100, balance_factor=0.1, 
                           norm='none', centroids=None):
    # Randomly initialize centroids by selecting k unique data points
    assert data.size(0) >= k
    indices = torch.randperm(data.size(0))[:k]
    #note: 1.2.2.1
    if norm == "l2":
        data = F.normalize(data, p=2, dim=1)
    elif norm == "layer":
        # warning: no trainable weight/bias applied
        data = F.layer_norm(data, data.shape[1:])
    else:
         pass
    if centroids is None:
        centroids = data[indices]  # k x D
    
    for _ in range(max_iter):
        # Compute cosine similarity between data points and centroids
        # data: N x 1 x D, centroids: 1 x k x D -> similarities: N x k
        similarities = F.cosine_similarity(data[:, None], centroids[None, :], dim=2)
        
        # Compute cluster sizes
        cluster_sizes = torch.bincount(similarities.argmax(dim=1), minlength=k)
        
        # Compute balancing factor for each cluster
        cluster_balancing_factor = balance_factor * (cluster_sizes.float() / data.size(0))
        
        # Assign clusters based on the highest cosine similarity and balancing factor
        adjusted_similarities = similarities - cluster_balancing_factor
        cluster_assignments = adjusted_similarities.argmax(dim=1)  # N x k -> N
        
        # Update centroids
        new_centroids = torch.stack([
            data[cluster_assignments == i].mean(dim=0)
            for i in range(k)
        ])
        
        # Check for convergence
        if torch.norm(centroids - new_centroids, dim=1).sum() < tol:
            break
        
        centroids = new_centroids
    
    return centroids, cluster_assignments


# create a balanced map from inv_map
@torch.no_grad()
def get_inverse_mapping(args, mapping):
    
    def _sort_ordered_dict_by_queue_length(ordered_dict):
        # Create a list of tuples (key, length of queue)
        sorted_items = sorted(ordered_dict.items(), key=lambda item: len(item[1]), reverse=True)
        
        # Build a new OrderedDict with keys sorted by the length of their associated queues
        sorted_dict = OrderedDict(sorted_items)
        return sorted_dict
    
    def _count_inv_map(inv_map):
        count = 0
        for k, v in inv_map.items():
            count += len(v)
        return count
    
    inv_map = OrderedDict()
    C = len(mapping)
    N = args.proxy_prompt_len
    
    # for each class
    for i, v in enumerate(mapping):
        if v not in inv_map:
            inv_map[v] = deque([i])
        else:
            inv_map[v].append(i)
        
    # sampling mapping from inv_map
    new_mapping = torch.zeros(len(mapping), dtype=torch.int)
    # for each prompt, each sample at least p = ceil(C/N), all classes must be sampled
    p = math.ceil(C/N)  # at least p samples for each prompt
    for n in range(args.proxy_prompt_len):
        count = 0
        bflag = False
        while count < p:
            if bflag:
                break
            inv_map = _sort_ordered_dict_by_queue_length(inv_map)
            for idx, (m, v) in enumerate(inv_map.items()):
                if len(v) == 0:
                    if idx == 0:
                        bflag = True
                    break
                new_mapping[v.pop()] = n
                count += 1
                if count >= p:
                    break
        #print("count {}, {} left".format(count, _count_inv_map(inv_map)))
    return new_mapping


# map N to M
@torch.no_grad()
def static_random_balanced_mapping(N, M):
    assert M < N, "M must be less than N"
    mapping = []
    for i in range(N // M):
        mapping.extend(range(M))
        
    for j in range(M*(i+1), N):
        mapping.append(j % M)
    return torch.tensor(mapping)

def measure_latency(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency: %.2f seconds" % latency)
        return result
    return wrapper

@torch.no_grad()
def generate_mapping(args, cls_mean_features, centroids):
    if not args.quiet_mode:
        print("Generate a new mapping...")
        
    start_time = time.time()
    if args.initial_mapping == 'kmeans':
        centroids, mapping = cosine_kmeans(cls_mean_features, args.proxy_prompt_len, 
                                    norm=args.kmeans_norm, centroids=centroids)
    elif args.initial_mapping == 'kmeans++':
        centroids, mapping = balanced_cosine_kmeans(cls_mean_features, 
                                    args.proxy_prompt_len, norm=args.kmeans_norm,
                                    centroids=centroids)
    else:
        raise ValueError("Invalid initial mapping method.")
    end_time = time.time()
    latency = (end_time - start_time) * 1000
    
    if not args.quiet_mode:
        print(f"Latency for kmeans: %.2f ms \n" % latency)
        print(OrderedDict(sorted(Counter(mapping.tolist()).items())))
    
    return mapping, centroids, latency


@torch.no_grad()
def generate_semantic_mapping(args, model, context):
    
    def get_img_dict(dataset):
        class_indices = {i: [] for i in range(len(dataset.classes))}
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
        return class_indices
    
    # load data
    print(colorstr('yellow', "Generating Semantic map... "))
    start = time.time()
    if args.dataset in _FGVC_CATALOG:
        init_data, _, _, num_class = create_fgvc_dataset(args, quiet=True)
    elif args.dataset in _VTAB_CATALOG:
        init_data, _, num_class = create_vtab_dataset(args, quiet=True)
    elif args.dataset in _TORCH_VISION_CATALOG:
        init_data, _, num_class = create_tv_dataset(args)
    else:
        raise ValueError("Dataset not supported.")
    
    assert num_class >= args.proxy_prompt_len, "Number of prompts must be less than number of classes."
    
    if not args.dataset in _TORCH_VISION_CATALOG:
        init_sampler = MappingSampler(args, init_data.get_img_dict(), init_data.get_num_imgs())
    else:
        init_sampler = MappingSampler(args, get_img_dict(init_data), len(init_data), has_path=False)
    
    init_loader = DataLoader(init_data,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              pin_memory=True,
                              sampler=init_sampler,
                              drop_last=False)
    if not args.quiet_mode:
        data_iterator = tqdm(init_loader,
            bar_format='{desc}{percentage:2.2f}% [{n_fmt}/{total_fmt},'
                        '{elapsed}{postfix}]',
            ncols=96, ascii=True, desc='[GPU:%d]: '
            % (args.gpu))
    else:
        data_iterator = init_loader
    
    # for 1.2.3.1
    vit_size = args.model_name.split('-')[1]
    global_pool = args.model_name.split('-')[0] == "mae"
    local_model = create_vit(vit_size=vit_size, args=args, 
                        num_classes=num_class, global_pool=global_pool)
    local_model.to('cuda')
    load_model_parameters(model, local_model)
    local_model.separate_qkv_params()
    local_model.eval()
    
    # initialize mapping
    for step, (images, labels) in enumerate(data_iterator):
        images = images.cuda(0, non_blocking=True)
        labels = labels.cuda(0, non_blocking=True)
        # compute output
        with context:
            local_model.forward_features(images, labels=labels) # B x 768

    cls_mean_features = local_model.get_cls_mean_feature()
    
    mapping = None
    centroids = None
    
    if args.initial_mapping != 'all_classes' and args.proxy_prompt_len > 0:
        mapping, centroids, _ = generate_mapping(args, cls_mean_features, None)
    
    vpt = None
    last_vpt = None

    del local_model
    latency = time.time() - start
    print(colorstr('yellow', "Generate map with {:.2f}s in total".format(latency)))
    
    return mapping, centroids, vpt, last_vpt
    
#########################################################################################