from typing import List, Callable, Any
from contextlib import nullcontext
from collections import UserDict
import logging

import torch

from torch import nn, Tensor
from torch.cuda.amp import GradScaler, autocast

from grad_cache_con_learning.context_managers import RandContext
from losses import SupConLoss, ConLoss

logger = logging.getLogger(__name__)


class GradCache:
    def __init__(
            self,
            model: nn.Module,
            chunk_size: int,
            loss_fn: Callable[..., Tensor],
            loss_type: str = "SupCon",
            fp16: bool = False,
            scaler: GradScaler = None,
    ):
        self.model = model
        for param in model.parameters():
            param.requires_grad_(True)
        self.chunk_size = chunk_size

        self.loss_fn = loss_fn
        self.loss_type = loss_type

        if self.loss_type != "SelfCon" and self.loss_type != "SupCon" and self.loss_type != "SimCLR":
            raise Exception("Loss not implemented")
        
        if self.loss_type == "SelfCon" and not isinstance(self.loss_fn, ConLoss):
            raise Exception("'loss_type' does not correspond to 'loss_fn'")
        if (self.loss_type == "SupCon" or self.loss_type == "SimCLR") and not isinstance(self.loss_fn, SupConLoss):
            raise Exception("'loss_type' does not correspond to 'loss_fn'")
        

        if fp16:
            assert scaler is not None, "mixed precision training requires a gradient scaler passed in"

        self.fp16 = fp16
        self.scaler = scaler

        self._get_input_tensors_strict = False

    def __call__(self, *args, **kwargs):
        return self.cache_step(*args, **kwargs)

    def split_inputs(self, model_input, chunk_size: int) -> List:
        if isinstance(model_input, Tensor):
            return list(model_input.split(chunk_size, dim=0))

        else:
            raise NotImplementedError(f'Model input split not implemented for type {type(model_input)}')

    def get_input_tensors(self, model_input) -> List[Tensor]:
        if isinstance(model_input, Tensor):
            return [model_input]

        elif isinstance(model_input, (list, tuple)):
            return sum((self.get_input_tensors(x) for x in model_input), [])

        elif isinstance(model_input, (dict, UserDict)):
            return sum((self.get_input_tensors(x) for x in model_input.values()), [])

        elif self._get_input_tensors_strict:
            raise NotImplementedError(f'get_input_tensors not implemented for type {type(model_input)}')

        else:
            return []

    def compute_loss(self, reps: Tensor, labels:Tensor = None, **loss_kwargs) -> Tensor:
        loss = self.loss_fn(reps, labels, **loss_kwargs)
        return loss

    def forward_no_grad(
            self,
            model: nn.Module,
            model_input: Tensor,
    ) -> [Tensor, List[RandContext]]:
        rnd_states = []
        model_reps = []

        with torch.no_grad():
            for x in model_input:
                rnd_states.append(RandContext(*self.get_input_tensors(x)))
                if self.loss_type == "SelfCon":
                    y1, y2 = model(x)
                    model_reps.append(torch.cat([f.unsqueeze(1) for f in y1] + [y2.unsqueeze(1)], dim=1))
                else:
                    features = model(x)
                    bsz = int(features.shape[0]/2)
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    model_reps.append(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1))
        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states

    def build_cache(self, reps: Tensor, labels: Tensor=None, **loss_kwargs) -> [List[Tensor], Tensor]:
        reps = reps.detach().requires_grad_()
        with autocast() if self.fp16 else nullcontext():
            loss = self.compute_loss(reps, labels, **loss_kwargs)

        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        cache = reps.grad

        return cache, loss.detach()

    def forward_backward(
            self,
            model: nn.Module,
            model_input,
            cached_gradients: List[Tensor],
            random_states: List[RandContext],
            no_sync_except_last: bool = False
    ):
        if no_sync_except_last:
            sync_contexts = [model.no_sync for _ in range(len(model_input) - 1)] + [nullcontext]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_input))]

        for x, state, gradient, sync_context in zip(model_input, random_states, cached_gradients, sync_contexts):
            with sync_context():
                if self.loss_type == "SelfCon":
                    with state:
                        y1, y2 = model(x)
                    reps = torch.cat([f.unsqueeze(1) for f in y1] + [y2.unsqueeze(1)], dim=1)
                else:
                    with state:
                        features = model(x)
                    bsz = int(features.shape[0]/2)
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    reps = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                reps = torch.cat([f.unsqueeze(1) for f in y1] + [y2.unsqueeze(1)], dim=1)
                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                surrogate.backward()

    def cache_step(
            self,
            model_input: Tensor,
            labels: Tensor=None,
            no_sync_except_last: bool = False,
            **loss_kwargs
    ) -> Tensor:

        model_input = self.split_inputs(model_input, self.chunk_size)

        model_reps, rnd_states = self.forward_no_grad(self.model, model_input)
        cache, loss = self.build_cache(model_reps, labels, **loss_kwargs)
        cache = cache.split(self.chunk_size)
        self.forward_backward(self.model, model_input, cache, rnd_states, no_sync_except_last=no_sync_except_last)
        return loss
