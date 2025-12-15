import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from contextlib import nullcontext
from collections import Counter


class VPTCriterion(torch.nn.Module):
    def __init__(self, args, embed_dim=768, mapping=None):
        """
        Module ProxyAnchor/NCA loss for ViT
        """
        super(VPTCriterion, self).__init__()

        self.args = args
        self.num_proxies = args.proxy_prompt_len
        self.num_prompts = args.proxy_prompt_len
        self.num_classes = args.num_classes
        self.embed_dim = embed_dim

        self.reset_num_batches()
        self.centroids = None
        
        self.mapping = None
        if mapping is not None:
            self.mapping = mapping.cpu().tolist()

        self.class_idxs = torch.arange(self.num_proxies)

        if self.args.initial_mapping == 'all_classes':
            self.class_idxs = torch.arange(self.num_classes)
            
        self.name = 'proxynca' if args.criterion == 'proxynca' else 'proxyanchor'
        self.proxy_dim = 1 if args.criterion == 'proxynca' else 0 # anchor: 0, nca: 1
    
    def set_centroids(self, centroids):
        if centroids is not None:
            self.centroids = centroids.to('cuda')
        else:
            self.centroids = None
    
    def get_centroids(self):
        return self.centroids
    
    def update_mapping(self, mapping):
        self.mapping = mapping.cpu().tolist()
    
    def reset_num_batches(self):
        self.num_batches = 0
        # for kmeans mapping
        self.num_class_batches = torch.zeros(self.num_classes, dtype=torch.long, device='cuda')
        # for init
        self.num_vpt_cls_batches = torch.zeros(self.num_classes, dtype=torch.long, device='cuda')
        self.num_prompt_batches = torch.zeros(self.num_prompts, dtype=torch.long, device='cuda')
        
    @torch.no_grad()
    def create_cls_mean_features(self, batch, labels, class_features):
        # batch: BS x dim
        # labels: torch.Tensor, list
        # return: num_classes x dim
        
        # TODO: add multi label case
        # 1. single label -> to tensor
        # 2. multi label -> iterate through batch
        #if isinstance(labels, list) and isinstance(labels[0], int):
        labels = torch.tensor(labels, device=batch.device)

        if isinstance(labels, torch.Tensor):
            labels = labels.long()
            labels = labels.view(-1)
            self._create_mean_from_tensor_labels(batch, labels, class_features)
        elif isinstance(labels[0], list):
            self._create_mean_from_list_labels(batch, labels, class_features)
        else:
            raise ValueError("Invalid label type")
    
    def _create_mean_from_tensor_labels(self, batch, labels, class_features):
        assert isinstance(labels, torch.Tensor)
        labels = labels.long().view(-1)
        unique_labels = labels.unique()
        for label in unique_labels:
            mask = labels == label
            label = int(label)
            class_values = batch[mask] # [num_samples, dim]
            class_features[label] = (class_features[label] * self.num_class_batches[label] + 
                        class_values.sum(0)) / (self.num_class_batches[label] + mask.sum())
            self.num_class_batches[label] += mask.sum()
    
    def _create_mean_from_list_labels(self, batch, labels, class_features):
        assert isinstance(labels, list)
        assert isinstance(labels[0], list)
        assert batch.shape[0] == len(labels)
        # multi labels
        for i, label in enumerate(labels):
            for l in label:
                class_features[l] = (class_features[l] * self.num_class_batches[l] +
                                     batch[i]) / (self.num_class_batches[l] + 1)
                self.num_class_batches[l] += 1
    
    
    
    
    # @torch.no_grad()
    # def create_class_aware_vpt_initial_in_batch(self, batch, class_vpt, attn, labels) -> None:
    #     # labels: BS
    #     # batch: BS x N x dim
    #     #?: class_vpt: num_class x dim
    #     # attn: Attention object
    #     # warning: should assume initial value is zero
    #     #assert self.num_batch > 0, "num_batch should be larger than 0"
    #     #self.num_mul_batches = self.num_mul_batches.to(batch.device)
        
    #     B, N, D = batch.size()
    #     top_k_indices, _ = self._select_indies(attn, topk=self.args.init_select_topk) # [B, topk]
    #     top_k_embeddings = torch.gather(batch[:, (1 + self.num_prompts):, :], dim=1, 
    #                         index=top_k_indices.unsqueeze(-1).expand(-1, -1, D))
    #     # B x topk x D -> B x D
    #     if self.args.init_pooling_type == 'mean':
    #         top_k_embeddings = top_k_embeddings.mean(dim=1) # [B, dim]
    #     else:
    #         top_k_embeddings = top_k_embeddings.max(dim=1)[0] # [B, dim]
        
    #     unique_labels = labels.unique()
    #     for label in unique_labels:
    #         mask = labels == label
    #         class_values = top_k_embeddings[mask] # [num_samples, dim]
    #         if self.args.init_pooling_type == 'mean':
    #             class_vpt[label] = (class_vpt[label] * self.num_vpt_cls_batches[label] + 
    #                     class_values.sum(0)) / (self.num_vpt_cls_batches[label] + mask.sum())
    #             self.num_vpt_cls_batches[label] += mask.sum()
    #         else:
    #             class_vpt[label] = torch.max(class_vpt[label], class_values.max(0)[0])
    
    # @torch.no_grad()
    # def create_last_vpt_initial_from_mapping(self, class_vpt, mapping):
    #     # vpt: num_prompts x dim
    #     # class_vpt: num_classes x dim
    #     # mapping: num_classes
        
    #     last_vpt = torch.zeros(self.proxy_prompt_len, self.embed_dim, device=class_vpt.device)
        
    #     if self.args.init_pooling_type == 'max':
    #         for cls, mapped_cls in enumerate(mapping):
    #             last_vpt[mapped_cls] = torch.max(last_vpt[mapped_cls], class_vpt[cls])
            
    #     else:
    #         for cls, mapped_cls in enumerate(mapping):
    #             last_vpt[mapped_cls] = (last_vpt[mapped_cls] * self.num_prompt_batches[mapped_cls] + 
    #                     class_vpt[cls]) / (self.num_prompt_batches[mapped_cls] + 1)
    #             self.num_prompt_batches[mapped_cls] += 1
    #     return last_vpt.unsqueeze(0)
        
    
    # @torch.no_grad()
    # def create_vpt_initial_in_batch(self, batch, vpt, attn) -> None:
    #     # batch: BS x N x dim
    #     # vpt: num_prompts x dim
    #     # attn: Attention object
    #     # warning: should assume initial value is zero
    #     #assert self.num_batch > 0, "num_batch should be larger than 0"
    #     B, N, D = batch.size()
    #     top_k_indices, _ = self._select_indies(attn, topk=self.args.init_select_topk) # [B, topk]
    #     top_k_embeddings = torch.gather(batch[:, (1 + self.num_prompts):, :], dim=1, 
    #                         index=top_k_indices.unsqueeze(-1).expand(-1, -1, D))
    #     # B x topk x D -> B*topk x D
    #     if self.args.init_pooling_type == 'mean':
    #         top_k_embeddings = top_k_embeddings.reshape(-1, D).mean(dim=0) # [dim]
    #     else:
    #         top_k_embeddings = top_k_embeddings.reshape(-1, D).max(dim=0)[0] # [dim]
    #     # num_prompts x dim
    #     top_k_embeddings = top_k_embeddings.unsqueeze(0).expand(self.num_prompts, -1)
        
    #     if self.args.init_pooling_type == 'mean':
    #         # param = (param * num_batches + value) / (num_batches + 1)
    #         vpt.mul_(self.num_batches).add_(top_k_embeddings).div_(self.num_batches + 1)
    #         self.num_batches += 1
    #     else:
    #         vpt.max_(top_k_embeddings)

    # @torch.no_grad()
    # def exponential_moving_avg(self, batch, vpt, attn) -> None:
    #     # batch: BS x N x dim
    #     # vpt: 1 x num_prompts x dim
    #     # attn: Attention object
    #     B, N, dim = batch.size()
    #     top_k_indices, _ = self._select_indies(attn, topk=self.args.ema_select_topk) # [B, topk]
    #     top_k_embeddings = torch.gather(batch[:, (1 + self.num_prompts):, :], dim=1, 
    #                         index=top_k_indices.unsqueeze(-1).expand(-1, -1, dim))
    #     # B x topk x dim
    #     if self.args.pooling_type == 'mean':
    #         top_k_embeddings = top_k_embeddings.mean(dim=1).mean(dim=0) # [dim]
    #     else:
    #         top_k_embeddings = top_k_embeddings.max(dim=1)[0].max(dim=0)[0] # [dim]
    #     top_k_embeddings = top_k_embeddings.unsqueeze(0).expand(self.num_prompts, -1).unsqueeze(0) # [1, num_prompts, dim]
    #     #vpt = vpt * self.args.ema_decay + top_k_embeddings * (1 - self.args.ema_decay)
    #     vpt.mul_(self.args.ema_decay).add_(top_k_embeddings * (1 - self.args.ema_decay))
    #     pass
    
    
    def fast_selection(self, batch, vpt, attn, labels, output, **kwargs):
        # select salient all patches for anchor loss
        # batch: BS x N x dim
        # labels: B
        # attn: Attention object
        # attn._qkv_emb     (before split) [B, N, 3*dim]
        # attn._q, attn._k, attn._v [B, N, dim]
        # vpt: 1 x num_prompts x dim
        # output: BS x N x dim
        
        B, N, dim = batch.size()
        query = attn._q # B x N x dim
        key = attn._k # B x N x dim
        
        # 1 x num_prompts x dim
        query_vpt = query[:, 1 : (self.num_proxies + 1), :].mean(dim=0, keepdim=True)
        key_vpt = key[:, 1 : (self.num_proxies + 1), :].mean(dim=0, keepdim=True)
        
        #! out-cls, should change label method
        out_vpt = output[:, 0, :].unsqueeze(0) # 1 x B x dim
        
        # cls token
        cls = batch[:, 0, :] # B x dim
        query_cls = query[:, 0, :] # B x dim
        key_cls = key[:, 0, :] # B x dim
        
        # patch
        patch = batch[:, (1 + self.num_proxies):, :].mean(dim=1, keepdim=False) # B x dim
        query_patch = query[:, (1 + self.num_proxies):, :].mean(dim=1, keepdim=False) # B x dim
        key_patch = key[:, (1 + self.num_proxies):, :].mean(dim=1, keepdim=False) # B x dim
        
        #! out-vpt, should select by labels
        out_patch = output[:, 1 : (self.num_proxies + 1), :] # B x 20 x dim
        
        # map labels to vpt idx
        # labels = labels.long()
        # labels = labels.view(-1) # BS*k
        labels = self._map_label_values(labels, self.mapping).to(batch.device) # BS
        
        # select out_patch by index in labels
        out_patch = torch.gather(out_patch, dim=1, 
                        index=labels.unsqueeze(1).unsqueeze(1).expand(-1, 1, dim)) # BS x 1 x dim
        
        out_patch = out_patch.squeeze(1) # BS x dim
        
        return (patch, query_patch, key_patch, out_patch), \
                (vpt, query_vpt, key_vpt, out_vpt), \
                (cls, query_cls, key_cls), \
                labels
        
    
    #note: allow gradient passing
    #! this is too slow
    # def select_salient_patches(self, batch, vpt, attn, labels, **kwargs):
        # select salient topk from 196 patches for anchor loss
        # batch: BS x N x dim
        # labels: B
        # attn: Attention object
        # attn._qkv_emb     (before split) [B, N, 3*dim]
        # attn._q, attn._k, attn._v [B, N, dim]
        # attn.full_attn (after softmax)    [B, num_heads, N, N]
        # vpt: 1 x num_prompts x dim
        #note N = 1[cls]+20[vpt]+196[patch] = 217
        
        B, N, dim = batch.size()
        num_heads = attn.full_attn.size()[1]
        
        # for 1.2.2.6
        query = attn._q.unsqueeze(0) # 1 x B x N x dim
        key = attn._k.unsqueeze(0) # 1 x B x N x dim
        
        # ===============================================================
        # 1 x num_prompts x dim
        query_vpt = query.squeeze(0)[:, 1 : (self.num_proxies + 1), :].mean(dim=0, keepdim=True) 
        key_vpt = key.squeeze(0)[:, 1 : (self.num_proxies + 1), :].mean(dim=0, keepdim=True)
        
        # cls token
        query_cls = query.squeeze(0)[:, 0, :] # B x dim
        cls = batch[:, 0, :] # B x dim
        
        key = key.squeeze(0) # B x N x dim
        
        #Note: select from multi-head attn_map
        # _att_map: [B, num_heads, topk] or [B, topk]
        top_k_indices, _att_map = self._select_indies(attn, topk=self.args.topk) # [B, topk] or [B, num_heads, topk]
        
        with torch.no_grad() if self.args.local_layer_gradient else nullcontext():
            
            if not self.args.key_patch_emb:
                sample = batch
            else:
                sample = key
                
            if self.args.select_head_composed_patch:
                # sample: B x N x dim -> B x N x num_heads x dim//num_heads
                # top_k_indices: B x num_heads x topk
                top_k_indices = top_k_indices.permute(0, 2, 1) # B x topk x num_heads
                sample = sample.unsqueeze(-2).view(B, N, num_heads, -1) # B x N x num_heads x dim//num_heads
                top_k_embeddings = torch.gather(sample[:, (1 + self.num_proxies):, :, :], dim=1,
                                index=top_k_indices.unsqueeze(-1).expand(-1, -1, -1, dim // num_heads)) # [B, topk, num_heads, dim//num_heads]
                
                _att_map = _att_map.permute(0, 2, 1) # B x topk x num_heads
                if self.args.attention_scaling_head:
                    _att_map = _att_map.softmax(dim=-1) # B x topk x num_heads
                if self.args.attention_scaling_topk and self.args.topk > 1:
                    _att_map = _att_map.softmax(dim=1) # B x topk x num_heads
                #note: make sure multiply _att_map for only 1 time
                if self.args.attention_scaling_head or self.args.attention_scaling_topk:
                    top_k_embeddings = (top_k_embeddings * _att_map.unsqueeze(-1))
                    
                top_k_embeddings = top_k_embeddings.view(B, -1, dim) # B x topk x dim
                sample = sample.view(B, -1, dim) # B x N x dim
                
            else:
                top_k_embeddings = torch.gather(sample[:, (1 + self.num_proxies):, :], dim=1, 
                                index=top_k_indices.unsqueeze(-1).expand(-1, -1, dim)) # [B, topk, dim]
                if self.args.attention_scaling_topk and self.args.topk > 1:
                    _att_map = _att_map.softmax(dim=1)
                    top_k_embeddings = (top_k_embeddings * _att_map.unsqueeze(-1))
            
                    
            if self.args.pooling_type == 'mean':
                top_k_embeddings = top_k_embeddings.mean(dim=1) # [B, dim]
                full_embeddings = sample.mean(dim=1) # [B, dim]
            else:
                top_k_embeddings = top_k_embeddings.max(dim=1)[0] # [B, dim]
                full_embeddings = sample.max(dim=1)[0] # [B, dim]
            
        #warning: labels = labels.unsqueeze(1).expand(-1, self.args.topk).reshape(-1) # BS*k
        
        top_k_embeddings = top_k_embeddings.view(-1, self.embed_dim) # BS x dim
        labels = labels.view(-1) # BS*k
        
        labels = self._map_tensor_values(labels, self.mapping) # BS
        
        ### ================================================================================== ###
        # return xs, ps, cls, labels
        return (top_k_embeddings, full_embeddings), (vpt, query_vpt, key_vpt), (cls, query_cls), labels
        
    
    def _dynamic_random_mapping(self, N, M):
        assert M < N, "M must be less than N"
        mapping = list(range(M))
        for i in range(M, N):
            mapping.append(random.randint(0, M - 1))
        return mapping
    
    
    def _select_most_frequent(self, tensor):
        assert isinstance(tensor, torch.Tensor)
        assert len(tensor.size()) == 1
        # Get the unique values and their counts
        unique_values, counts = torch.unique(tensor, return_counts=True)
        # Find the maximum frequency
        max_count = torch.max(counts)
        # Find the indices of the values with the maximum frequency
        most_frequent_indices = (counts == max_count).nonzero(as_tuple=True)[0]
        # Randomly choose one of the most frequent values if there are ties
        chosen_index = random.choice(most_frequent_indices.tolist())
        return unique_values[chosen_index].item()
    
    
    def _map_label_values(self, labels, mapping):
        # transform data labels to mapped labels
        if isinstance(labels, torch.Tensor):
            mapped_labels = labels.clone()
            for i in range(len(mapping)):
                mapped_labels[labels == i] = mapping[i]
            return mapped_labels
        else:
            # 2d list
            # select most frequent label
            mapped_labels = torch.zeros(len(labels), dtype=torch.long)
            mapping_labels = torch.tensor(mapping, dtype=torch.long)
            for i, label_list in enumerate(labels):
                mapped_labels[i] = self._select_most_frequent(mapping_labels[label_list])
            return mapped_labels
    
    # def refactorization(self, vpt, labels, **kwargs):
        # vpt: 1 x num_prompts x dim
        # labels: BS
        # self.factor: num_prompts x num_classes
        
        M = self.factor.unsqueeze(0).expand(len(labels), -1, -1) # BS x num_prompts x num_classes
        labels = labels.unsqueeze(1).expand(-1, self.num_prompts).unsqueeze(-1) # BS x num_prompts x 1
        selected_M = torch.gather(M, dim=-1, index=labels).squeeze(-1) # BS x num_prompts
        # warning: filter out negative scaling ? or gate ?
        if self.args.ref_active: 
            selected_M = self.relu(selected_M) # BS x num_prompts
        elif self.args.ref_active == 'tanh':
            selected_M = torch.tanh(selected_M)
        else:
            pass
        
        dia_M = torch.diag_embed(selected_M) # BS x num_prompts x num_prompts
        if self.args.ref_pooling == 'mean':
            refactored_vpt = (dia_M @ vpt).mean(dim=1) # BS x dim 
        elif self.args.ref_pooling == 'max':
            refactored_vpt = (dia_M @ vpt).max(dim=1)[0] # BS x dim 
        else:
            raise ValueError(f"Invalid pooling type: {self.args.ref_pooling}")
        
        return refactored_vpt
    
    
    def forward(self, batch, vpt, labels, attn, output, **kwargs):
        
        patch_loss = torch.Tensor([0.0]).to(batch.device)
        cls_loss = torch.Tensor([0.0]).to(batch.device)
        # warning: self.class_idxs = torch.arange(self.num_proxies)
        # xs[2], ps[3], cls[2], labels[1]
        #xs, ps, cls, labels = self.select_salient_patches(batch, vpt,
        #                        attn, labels, **kwargs) # BS*k x dim
        
        if isinstance(labels, list) and isinstance(labels[0], int):
            labels = torch.tensor(labels, device=batch.device)
        
        if isinstance(labels, torch.Tensor):
            labels = labels.long()
            labels = labels.view(-1)
        
        xs, ps, cls, labels = self.fast_selection(batch, vpt,
                            attn, labels, output, **kwargs) # BS*k x dim
        #labels = labels.to(batch.device)
        
        if not self.args.test_vpt_loss_off:
            patch_loss = self.ml_patch_forward(xs, ps, labels, **kwargs)
        
        if not self.args.test_cls_loss_off:
            cls_loss = self.ml_cls_forward(cls, ps, labels, **kwargs)
        
        return self.args.vpt_loss_weight * patch_loss \
            + self.args.cls_loss_weight * cls_loss
            
    
    def ml_cls_forward(self, clses, vpts, labels, **kwargs):
        # (cls, query_cls), (vpt, query_vpt, key_vpt)
        # clses[3]: BS x dim
        # vpts[4]: 1 x num_prompts x dim
        # labels: # BS value: [0, num_prompts-1]
        # warning: self.class_idxs = torch.arange(self.num_proxies)
        # Note: normalize should be after selection
        # if self.args.query_cls_emb:
        #     cls = clses[1]
        # else:
        #     cls = clses[0]
        # if self.args.key_vpt_emb:
        #     vpt = vpts[2]
        # else:
        #     vpt = vpts[0]
        
        if self.args.cls_type == 'cls':
            cls = clses[0]
        elif self.args.cls_type == 'query':
            cls = clses[1]
        elif self.args.cls_type == 'key':
            cls = clses[2]
        else:
            raise ValueError(f"Invalid cls type: {self.args.cls_type}")

        if self.args.vpt_type_for_cls_ml == 'vpt':
            vpt = vpts[0]
        elif self.args.vpt_type_for_cls_ml == 'query':
            vpt = vpts[1]
        elif self.args.vpt_type_for_cls_ml == 'key':
            vpt = vpts[2]
        #elif self.args.vpt_type_for_cls_ml == 'output':
        #    vpt = vpts[3]
        else:
            raise ValueError(f"Invalid vpt type: {self.args.vpt_type}")
        
        
        device = cls.device
        
        cls_batch = F.normalize(cls, dim=-1) # [cls] token BS x dim
        proxies = F.normalize(vpt.squeeze(0), dim=-1) * self.args.loss_vpt_scale_ratio # num_prompts x dim
        
        self.labels = labels.unsqueeze(1) # BS x 1
        self.f_labels = self.labels.view(-1) # BS  flat labels
        self.class_idxs = self.class_idxs.to(device) # num_prompts
        
        self.same_labels = (self.class_idxs.unsqueeze(1) == self.labels.T).to(
                device).T # BS x num_prompts  positive mask: one-hot labels
        self.diff_labels = (self.class_idxs.unsqueeze(1) != self.labels.T).to(
                device).T # BS x num_prompts  negative mask: invert of one-hot labels
        
        sims = cls_batch.mm(proxies.T) # BS x num_prompts
        
        w_pos_sims = -self.args.loss_oproxy_pos_alpha * (sims - self.args.loss_oproxy_pos_delta)
        w_neg_sims = self.args.loss_oproxy_neg_alpha * (sims - self.args.loss_oproxy_neg_delta)
        
        pos_s = self.masked_logsumexp(w_pos_sims,
                                      mask=self.same_labels.type(torch.bool),
                                      dim=self.proxy_dim)
        neg_s = self.masked_logsumexp(w_neg_sims,
                                      mask=self.diff_labels.type(torch.bool),
                                      dim=self.proxy_dim)
        return pos_s.mean() + neg_s.mean()
    
        
    def ml_patch_forward(self, patchs, vpts, labels, **kwargs):    
        # (top_k_embeddings, full_embeddings), (vpt, query_vpt, key_vpt)
        # patchs[4]: BS x dim
        # vpts[4]: 1 x num_prompts x dim
        # labels: # BS value: [0, num_prompts-1]
        
        # =============================== DML ============================================== #
        
        # Note: normalize should be after selection
        if self.args.patch_type == 'patch':
            batch = patchs[0]
        elif self.args.patch_type == 'query':
            batch = patchs[1]
        elif self.args.patch_type == 'key':
            batch = patchs[2]
        elif self.args.patch_type == 'output':
            batch = patchs[3]
        else:
            raise ValueError(f"Invalid patch type: {self.args.patch_type}")
        
        if self.args.vpt_type_for_patch_ml == 'vpt':
            vpt = vpts[0]
        elif self.args.vpt_type_for_patch_ml == 'query':
            vpt = vpts[1]
        elif self.args.vpt_type_for_patch_ml == 'key':
            vpt = vpts[2]
        else:
            raise ValueError(f"Invalid vpt type: {self.args.vpt_type}")

        batch = F.normalize(patchs[0], dim=-1)  
        proxies = F.normalize(vpt.squeeze(0), dim=-1) * self.args.loss_vpt_scale_ratio # num_prompts x dim
        
        self.labels = labels.unsqueeze(1) # BS*k x 1
        self.f_labels = self.labels.view(-1) # BS*k  flat labels
        
        self.class_idxs = self.class_idxs.to(batch.device) # mapped idx, num_prompts
        
        self.same_labels = (self.class_idxs.unsqueeze(1) == self.labels.T).to(
                batch.device).T # BS*k x num_prompts  positive mask: one-hot labels
        self.diff_labels = (self.class_idxs.unsqueeze(1) != self.labels.T).to(
                batch.device).T # BS*k x num_prompts  negative mask: invert of one-hot labels
        
        sims = None
        sims = batch.mm(proxies.T) # BS*k x num_prompts
        
        # # BS x num_prompts
        # if self.args.full_patch_positive:
        #     w_pos_sims = -self.args.loss_oproxy_pos_alpha * (full_sims - self.args.loss_oproxy_pos_delta)
        # else:
        #     w_pos_sims = -self.args.loss_oproxy_pos_alpha * (sims - self.args.loss_oproxy_pos_delta)
        # if self.args.full_patch_negative:
        #     w_neg_sims = self.args.loss_oproxy_neg_alpha * (full_sims - self.args.loss_oproxy_neg_delta)
        # else:
        #     w_neg_sims = self.args.loss_oproxy_neg_alpha * (sims - self.args.loss_oproxy_neg_delta)
        
        w_pos_sims = -self.args.loss_oproxy_pos_alpha * (sims - self.args.loss_oproxy_pos_delta)
        w_neg_sims = self.args.loss_oproxy_neg_alpha * (sims - self.args.loss_oproxy_neg_delta)
        
        pos_s = self.masked_logsumexp(w_pos_sims,
                                      mask=self.same_labels.type(torch.bool),
                                      dim=self.proxy_dim)
        neg_s = self.masked_logsumexp(w_neg_sims,
                                      mask=self.diff_labels.type(torch.bool),
                                      dim=self.proxy_dim)
        return pos_s.mean() + neg_s.mean()
    
    
    def masked_logsumexp(self, sims, dim=0, mask=None):
        # select features by mask for proxy DML
        # fill false to -inf
        if mask is not None:
            sims = sims.masked_fill(~mask, torch.finfo(sims.dtype).min)
        dims = list(sims.shape) # [BS*k, num_proxies]
        # create [BS*k, 1] or [1, num_proxies] zeros
        # append zeros to the last index of dimension 0 or 1
        # because exp(0) = 1, this equal to adding 1 to the sum_exp()
        dims[dim] = 1 # select between nca and anchor loss
        zeros = torch.zeros(dims, dtype=sims.dtype, device=sims.device)
        sims = torch.cat([sims, zeros], dim=dim)
        logsumexp_sims = torch.logsumexp(sims, dim=dim, keepdim=True)
        # fill -inf back to 0
        if mask is not None:
            logsumexp_sims = logsumexp_sims.masked_fill(
                ~torch.any(mask, dim=dim, keepdim=True), 0)
        return logsumexp_sims
    
    
    # @torch.no_grad()
    # def _select_indies(self, attn, topk=1):
    #     # attn.full_attn: [B, num_heads, 217, 217]
    #     B = attn.full_attn.size(0)
    #     if self.args.select_type == 'cross_attn':
    #         attn_map = attn.full_attn[:, :, (1 + self.num_proxies):, 
    #                                   (1 + self.num_proxies):] # [B, num_heads, 196, 196]
    #         mask = (1 - torch.eye(attn_map.size(-1))).to(attn_map.device) # [196, 196]
    #         attn_map = attn_map * mask.expand(B, attn.num_heads, -1, -1) # remove diagonal
    #         attn_map = attn_map.sum(dim=3).squeeze(-1) # [B, num_heads, 196]
            
    #     elif self.args.select_type == 'cls_attn':
    #         attn_map = attn.full_attn[:, :, 0, (1 + self.num_proxies):] # [B, num_heads, 1, 196]
    #         attn_map = attn_map.squeeze(2) # [B, num_heads, 196]
    #     else:
    #         raise ValueError(f"Invalid attention type: {self.args.attn_type}")
        
    #     if not self.args.select_head_composed_patch:
    #         if self.args.head_pooling == 'mean':
    #             attn_map = attn_map.mean(dim=1)
    #         elif self.args.head_pooling == 'max':
    #             attn_map = attn_map.max(dim=1)[0]
    #         else:
    #             raise ValueError(f"Invalid head pooling type: {self.args.head_pooling}")
    #         _map, top_k_indices = torch.topk(attn_map, topk, dim=-1) # [B, topk]
    #     else:
    #         # attn_map: [B, num_heads, 196]
    #         _map, top_k_indices = torch.topk(attn_map, topk, dim=-1) # [B, num_heads, topk]
              
    #     return top_k_indices, _map

        