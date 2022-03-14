# MoCo-related code is modified from https://github.com/facebookresearch/moco
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../')
from backbone.select_backbone import select_backbone


def cascade_knn(anchor_main, anchor_aux, m_bank_main, m_bank_aux, index_record, anchor_index_mask, topk, ratio, stage):
    m_bank_main = m_bank_main.T
    m_bank_aux =  m_bank_aux.T

    batchSize = anchor_main.size(0)
    outSize = m_bank_main.size(0)
    inputSize = m_bank_main.size(1)

    top_pos = topk
    num_stage = stage
    select_ratio = ratio
    index_record_all_stage = []
    for stage_idx in range(num_stage):
        # Cascade Top k for each stage.
        cas_topk = int(outSize * select_ratio ** (stage_idx+1))
        if stage_idx == 0:
            topk_start=cas_topk
            #------ stage 1 ------#
            # calcualte similarity between anchor_m1 and its memeory bank
            sim_scores_start = torch.einsum("nd,md->nm", anchor_main, m_bank_main.clone().detach())
            # mask the index of anchor itself
            sim_scores_start[anchor_index_mask] = -np.inf
            # select indexes of top-k most similar instances
            _, topk_idx_start = torch.topk(sim_scores_start, cas_topk, dim=1)
            # select features from both memory banks via selected index at stage 1
            nn_feats_start_aux = torch.index_select(m_bank_aux, dim=0, index=topk_idx_start.view(-1))
            nn_feats_start_aux = nn_feats_start_aux.view(batchSize, cas_topk, inputSize)
            nn_feats_start_main = torch.index_select(m_bank_main, dim=0, index=topk_idx_start.view(-1))
            nn_feats_start_main = nn_feats_start_main.view(batchSize, cas_topk, inputSize)
            # record root index from original memoery bank.
            index_record_start = torch.index_select(index_record, dim=0, index=topk_idx_start.view(-1))
            index_record_start = index_record_start.view(batchSize, cas_topk, 1)
            index_record_start = torch.cat((index_record_start,index_record_start[...,0:1]),2)
            index_record_start[...,1] = torch.arange(0, cas_topk, dtype=torch.int)
            index_record_all_stage.append(index_record_start)

        elif stage_idx < (num_stage - 1):
            if stage_idx % 2 != 0:
                if stage_idx == 1:
                    nn_feats_aux = nn_feats_start_aux
                    nn_feats_main = nn_feats_start_main
                    curr_nn_feats = nn_feats_aux
                    pre_index_record = index_record_start
                    anchors  = anchor_aux
                else:
                    anchors  = anchor_aux
                    curr_nn_feats = nn_feats_aux
            else:
                anchors = anchor_main
                curr_nn_feats = nn_feats_main
            # ------ interval stages with different modalities ------#
            # calcualte similarity between anchor_m2 and its memeory bank
            sim_scores = torch.bmm(curr_nn_feats, anchors.view(batchSize, inputSize, 1))
            # select index of top-k most similar instances
            _, topk_idx = torch.topk(sim_scores, cas_topk, dim=1)
            # select features from remaining features via selected index at stage 2
            nn_feats_main_list = []
            nn_feats_aux_list = []
            for batch_index in range(batchSize):
                one_bs_nn_main_feats = torch.index_select(nn_feats_main[batch_index,...], dim=0, index=topk_idx[batch_index,...].view(-1))
                nn_feats_main_list.append(one_bs_nn_main_feats)
                one_bs_nn_aux_feats = torch.index_select(nn_feats_aux[batch_index,...], dim=0, index=topk_idx[batch_index,...].view(-1))
                nn_feats_aux_list.append(one_bs_nn_aux_feats)
            nn_feats_main = torch.stack(nn_feats_main_list)
            nn_feats_aux= torch.stack(nn_feats_aux_list)

            # record root index from original memoery bank.
            curr_index_record_list = []
            for batch_index in range(batchSize):
                one_bs_nn_curr_index_record = torch.index_select(pre_index_record[batch_index,...], dim=0, index=topk_idx[batch_index,...].view(-1))
                curr_index_record_list.append(one_bs_nn_curr_index_record)
            curr_index_record = torch.stack(curr_index_record_list)
            curr_index_record = torch.cat((curr_index_record,curr_index_record[...,0:1]),2)
            curr_index_record[...,stage_idx+1] = torch.arange(0, cas_topk, dtype=torch.int)
            index_record_all_stage.append(curr_index_record)
            pre_index_record = curr_index_record

        elif stage_idx == (num_stage - 1):
            anchors = anchor_main
            curr_nn_feats = nn_feats_main
            #------ stage final ------#
            # calcualte similarity between anchor_m1 and its memeory bank
            sim_scores_end = torch.bmm(curr_nn_feats, anchors.view(batchSize, inputSize, 1))
            # select index of top-k most similar instances
            _, topk_idx_end = torch.topk(sim_scores_end, top_pos, dim=1)
            # record root index from original memoery bank.
            index_record_end_list = []
            for batch_index in range(batchSize):
                one_nearest_index_record_end = torch.index_select(pre_index_record[batch_index,...], dim=0, index=topk_idx_end[batch_index,...].view(-1))
                index_record_end_list.append(one_nearest_index_record_end)
            index_record_end = torch.stack(index_record_end_list)
            index_record_end = torch.cat((index_record_end,index_record_end[...,0:1]),2)
            index_record_end[...,stage_idx+1] = torch.arange(0, top_pos, dtype=torch.int)
            index_record_all_stage.append(index_record_end)

            # select remaining features at stage 1 as subset of training set.
            nn_from_m_bank_main = torch.index_select(m_bank_main, dim=0, index=topk_idx_start.view(-1))
            nn_from_m_bank_main = nn_from_m_bank_main.view(batchSize, topk_start, inputSize)

            pos_instance_index = index_record_end[...,0].long()

        pos_weights = torch.ones(batchSize, top_pos)

    return pos_instance_index, index_record_all_stage, pos_weights

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class InfoNCE(nn.Module):
    '''
    Basically, it's a MoCo for video input: https://arxiv.org/abs/1911.05722
    '''
    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(InfoNCE, self).__init__()

        self.dim = dim
        self.K = K
        self.m = m
        self.T = T

        # create the encoders (including non-linear projection head: 2 FC layers)
        backbone, self.param = select_backbone(network)
        feature_size = self.param['feature_size']
        self.encoder_q = nn.Sequential(
                            backbone,
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))

        backbone, _ = select_backbone(network)
        self.encoder_k = nn.Sequential(
                            backbone,
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Notes: for handling sibling videos, e.g. for UCF101 dataset


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        '''Momentum update of the key encoder'''
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        '''
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        '''
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, block):
        '''Output: logits, targets'''
        (B, N, *_) = block.shape # [B,N,C,T,H,W]
        assert N == 2
        x1 = block[:,0,:].contiguous()
        x2 = block[:,1,:].contiguous()
        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k = k.view(B, self.dim)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: B,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k)

        return logits, labels


class UberNCE(InfoNCE):
    '''
    UberNCE is a supervised version of InfoNCE,
    it uses labels to define positive and negative pair
    Still, use MoCo to enlarge the negative pool
    '''
    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(UberNCE, self).__init__(network, dim, K, m, T)
        # extra queue to store label
        self.register_buffer("queue_label", torch.ones(K, dtype=torch.long) * -1)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_label[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, block, k_label):
        '''Output: logits, binary mask for positive pairs
        '''
        (B, N, *_) = block.shape # [B,N,C,T,H,W]
        assert N == 2
        x1 = block[:,0,:].contiguous()
        x2 = block[:,1,:].contiguous()

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k = k.view(B, self.dim)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: B,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # mask: binary mask for positive keys
        mask = k_label.unsqueeze(1) == self.queue_label.unsqueeze(0) # B,K
        mask = torch.cat([torch.ones((mask.shape[0],1), dtype=torch.long, device=mask.device).bool(),
                          mask], dim=1) # B,(1+K)
        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k, k_label)
        return logits, mask


class CoCLR(InfoNCE):
    '''
    CoCLR: using another view of the data to define positive and negative pair
    Still, use MoCo to enlarge the negative pool
    '''
    def __init__(self, network='s3d', dim=128, K=2048, m=0.999, T=0.07, topk=5, ratio=0.5, stage=3, reverse=False, cascade=False):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(CoCLR, self).__init__(network, dim, K, m, T)

        self.topk = topk
        self.ratio = ratio
        self.stage = stage

        # create another encoder, for the second view of the data
        backbone, _ = select_backbone(network)
        feature_size = self.param['feature_size']
        self.sampler = nn.Sequential(
                            backbone,
                            nn.AdaptiveAvgPool3d((1,1,1)),
                            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
                            nn.ReLU(),
                            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))
        for param_s in self.sampler.parameters():
            param_s.requires_grad = False  # not update by gradient

        # create another queue, for the second view of the data
        self.register_buffer("queue_second", torch.randn(dim, K))
        self.queue_second = nn.functional.normalize(self.queue_second, dim=0)

        # for handling sibling videos, e.g. for UCF101 dataset
        self.register_buffer("queue_vname", torch.ones(K, dtype=torch.long) * -1)
        # for monitoring purpose only
        self.register_buffer("queue_label", torch.ones(K, dtype=torch.long) * -1)
        # for monitoring purpose only
        self.register_buffer("memory_index_record", torch.arange(0, K, dtype=torch.long))

        self.queue_is_full = False
        self.reverse = reverse
        self.cascade = cascade

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_second, vnames):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        keys_second = concat_all_gather(keys_second)
        vnames = concat_all_gather(vnames)
        # labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_second[:, ptr:ptr + batch_size] = keys_second.T
        self.queue_vname[ptr:ptr + batch_size] = vnames
        self.queue_label[ptr:ptr + batch_size] = torch.ones_like(vnames)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, block1, block2, k_vsource):
        '''Output: logits, targets'''
        (B, N, *_) = block1.shape # B,N,C,T,H,W
        assert N == 2
        x1 = block1[:,0,:].contiguous()
        f1 = block1[:,1,:].contiguous()
        x2 = block2[:,0,:].contiguous()
        f2 = block2[:,1,:].contiguous()

        if self.reverse:
            x1, f1 = f1, x1
            x2, f2 = f2, x2

        # compute query features
        q = self.encoder_q(x1)  # queries: B,C,1,1,1
        q = nn.functional.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k = k.view(B, self.dim)

            # compute key feature for second view
            kf = self.sampler(f2) # keys: B,C,1,1,1
            kf = nn.functional.normalize(kf, dim=1)
            kf = kf.view(B, self.dim)

        # if queue_second is full: compute mask & train CoCLR, else: train InfoNCE

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: N,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # mask: binary mask for positive keys
        # handle sibling videos, e.g. for UCF101. It has no effect on K400
        mask_source = k_vsource.unsqueeze(1) == self.queue_vname.unsqueeze(0) # B,K
        mask = mask_source.clone()

        if not self.queue_is_full:
            self.queue_is_full = torch.all(self.queue_label != -1)
            if self.queue_is_full: print('\n===== queue is full now =====')

        if self.queue_is_full and (self.topk != 0):
            mask_sim = kf.matmul(self.queue_second.clone().detach())
            mask_sim[mask_source] = - np.inf # mask out self (and sibling videos)
            if self.cascade:
                topk_onehot = torch.zeros_like(mask_sim)
                l_to_l_pos_index, l_to_l_index_record, pos_weights_l = cascade_knn(kf, q, self.queue_second,
                                                                                self.queue,
                                                                                self.memory_index_record,
                                                                                mask_source,
                                                                                self.topk,
                                                                                self.ratio,
                                                                                self.stage)
                topk_onehot.scatter_(1, l_to_l_pos_index, 1)
                mask[topk_onehot.bool()] = 1.0
            else:
                _, topkidx = torch.topk(mask_sim, self.topk, dim=1)
                topk_onehot = torch.zeros_like(mask_sim)
                topk_onehot.scatter_(1, topkidx, 1)
                mask[topk_onehot.bool()] = 1.0

        mask = torch.cat([torch.ones((mask.shape[0],1), dtype=torch.float, device=mask.device),
                mask.to(torch.float)], dim=1)

        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k, kf, k_vsource)

        return logits, mask.detach()
