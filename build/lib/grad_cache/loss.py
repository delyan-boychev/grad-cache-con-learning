from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch import distributed as dist


class SimpleContrastiveLoss:
    def __init__(self, n_hard_negatives: int = 0):
        self.target_per_qry = n_hard_negatives + 1

    def __call__(self, x: Tensor, y: Tensor, target: Tensor = None, reduction: str = 'mean'):
        if target is None:
            assert x.size(0) * self.target_per_qry == y.size(0)
            target = torch.arange(0, x.size(0) * self.target_per_qry, self.target_per_qry, device=x.device)

        logits = torch.matmul(x, y.transpose(0, 1))
        return F.cross_entropy(logits, target, reduction=reduction)

class ConLoss(nn.Module):
    """Self-Contrastive Learning: https://arxiv.org/abs/2106.15499."""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, supcon_s=False, selfcon_s_FG=False, selfcon_m_FG=False):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            supcon_s: boolean for using single-viewed batch.
            selfcon_s_FG: exclude contrastive loss when the anchor is from F (backbone) and the pairs are from G (sub-network).
            selfcon_m_FG: exclude contrastive loss when the anchor is from F (backbone) and the pairs are from G (sub-network).
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0] if not selfcon_m_FG else int(features.shape[0]/2)
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)    
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        if not selfcon_s_FG and not selfcon_m_FG:
            contrast_count = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            if self.contrast_mode == 'one':
                anchor_feature = features[:, 0]
                anchor_count = 1
            elif self.contrast_mode == 'all':
                anchor_feature = contrast_feature
                anchor_count = contrast_count
            else:
                raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        elif selfcon_s_FG:
            contrast_count = features.shape[1]
            anchor_count = features.shape[1]-1
            
            anchor_feature, contrast_feature = torch.cat(torch.unbind(features, dim=1)[:-1], dim=0), torch.unbind(features, dim=1)[-1]
            contrast_feature = torch.cat([anchor_feature, contrast_feature], dim=0)
        elif selfcon_m_FG:
            contrast_count = int(features.shape[1] * 2)
            anchor_count = (features.shape[1]-1)*2
            
            anchor_feature, contrast_feature = torch.cat(torch.unbind(features, dim=1)[:-1], dim=0), torch.unbind(features, dim=1)[-1]
            contrast_feature = torch.cat([anchor_feature, contrast_feature], dim=0)
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
            
        mask = mask * logits_mask
        if supcon_s:
            idx = mask.sum(1) != 0
            mask = mask[idx, :]
            logits_mask = logits_mask[idx, :]
            logits = logits[idx, :]
            batch_size = idx.sum()
            
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
            
        return loss

class DistributedContrastiveLoss(SimpleContrastiveLoss):
    def __init__(self, n_hard_negatives: int = 0):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."

        super().__init__(n_hard_negatives=n_hard_negatives)
        self.word_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def __call__(self, x: Tensor, y: Tensor, **kwargs):
        dist_x = self.gather_tensor(x)
        dist_y = self.gather_tensor(y)

        return super().__call__(dist_x, dist_y, **kwargs)

    def gather_tensor(self, t):
        gathered = [torch.empty_like(t) for _ in range(self.word_size)]
        dist.all_gather(gathered, t)
        gathered[self.rank] = t
        return torch.cat(gathered, dim=0)


class ContrastiveLossWithQueryClosure(SimpleContrastiveLoss):
    def __call__(
            self,
            *reps: Tensor,
            query_closure: Callable[[], Tensor] = None,
            target: Tensor = None,
            reduction: str = 'mean'
    ):
        if len(reps) == 0 or len(reps) > 2:
            raise ValueError(f'Expecting 1 or 2 tensor input, got {len(reps)} tensors')

        # no closure evaluation
        if len(reps) == 2:
            assert query_closure is None, 'received 2 representation tensors while query_closure is also set'
            return super().__call__(*reps, target=target, reduction=reduction)

        # run the closure
        assert query_closure is not None
        x = query_closure()
        y = reps[0]
        return super().__call__(x, y, target=target, reduction=reduction)
