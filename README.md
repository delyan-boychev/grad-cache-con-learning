# Gradient Cache Contrastive Learning
Gradient Cache Contrastive Learning is a technique for unlimitedly scaling contrastive learning batch far beyond GPU/TPU memory constraint in Computer Vision. This means training that used to take heavy hardware, e.g. 8 V100 GPU, can be done on a single GPU. In addition, Gradient Cache allow users to replace big RAM GPU/TPU with much more cost efficient high FLOP low RAM systems. It is an adopted version of the paper [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983) for the SimCLR, SupCon and SelfCon losses.
## Installation
First install Pytorch.  To install `grad-cache-con-learning`, run the following:
```sh
pip install grad_cache_con_learning
```

## Usage
Gradient caching functionalities are implemented in `GradCache` class.

### Initialization
The class's `__init__` method defines the cache and has a functional parameter `loss_fn` to flexibly set your loss function. 
```py
grad_cache.GradCache(  
  model: nn.Module,  
  chunk_size: int,
  loss_fn: Callable[..., Tensor],
  loss_type: str = "SupCon",
  fp16: bool = False,
  scaler: GradScaler = None, 
)
``` 
**model** - The encoder model to be updated with with the Gradient Cache.

**chunk_size** - An integer indicating chunk size. This controls the sub-batch size to run forward-backward pass and should be set based on available GPU memory. A value too small will leave the GPU under utilized.

**loss_fn** -  A loss function that takes representation tensors. It should compute the loss of the model based on the representations. The options are `grad_cache_con_learning.losses.SupConLoss` for SimCLR and SupCon, `grad_cache_con_learning.losses.ConLoss` for SelfCon.

**loss_type** - The loss type: 'SimCLR', 'SupCon' or 'SelfCon'.

**fp16** - If True, run mixed precision training, which requires scaler to also be set.

**scaler** - A GradScaler object for automatic mixed precision training.

### Cache Gradient Step
To run a cached gradient computatoin step, call `cache_step` function,

```py
cache_step(  
  model_input,
  model_input: Tensor,
  labels: Tensor = None,  
  no_sync_except_last: bool = False,  
  **loss_kwargs  
)
```
Run a single gradient cache step. Upon function return, updates are computed for each model in `self.models` with gradient populated on the weights, as if the `model_inputs` are run as a huge single batch on sufficiently large hardware.  Calling an GradCache object with `__call__` will also invoke this function.

**model_input** - Tensor which is the input for the encoder model.
**labels** -  Tensor which contains the true labels for training. For SimCLR we do not provide labels.

**no_sync_except_last** - If True, under distributed setup, for each model, only trigger gradient reduction across processes for the last sub-batch's forward-backward pass. This could come in handy when dealing with a) large model, and/or b) non trivial number of sub-batches.

**loss_kwargs** - Additional keyword arguments to the loss function `loss_fn`.

**Return** - loss, the current steps loss scaler tensor (detached from the graph).

## Example Usage with Contastive Losses (SimCLR, SupCon, SelfCon)
You need to preserve the original training procedure from the methods - [SimCLR and SupCon](https://github.com/HobbitLong/SupContrast), [SelfCon](https://github.com/raymin0223/self-contrastive-learning). It works only with the original methods.

### SupCon Example
First, you need to initialize the GradCache object,
```py
from grad_cache_con_learning import GradCache
from grad_cache_con_learning.losses import SupConLoss
...
loss_fn = SupConLoss()
gc = GradCache(
  model=model, 
  chunk_sizes=2, 
  loss_fn=loss_fn,
  loss_type="SupCon"
)
...
```
Only replace:
```py
...
optimizer.zero_grad()
features = model(images)
f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
loss = criterion(features, y)
loss.backward()
optimizer.step()
...
```
with the following:
```py
...
optimizer.zero_grad()
gc(x, y)
optimizer.step()
...
```
### SimCLR Example
First, you need to initialize the GradCache object,
```py
from grad_cache_con_learning import GradCache
from grad_cache_con_learning.losses import SupConLoss
...
loss_fn = SupConLoss()
gc = GradCache(
  model=model, 
  chunk_size=2, 
  loss_fn=loss_fn,
  loss_type="SimCLR" 
)
...
```
Only replace:
```py
...
optimizer.zero_grad()
features = model(images)
f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
loss = criterion(features)
loss.backward()
optimizer.step()
...
```
with the following:
```py
...
optimizer.zero_grad()
gc(images)
optimizer.step()
...
```
### SelfCon Example
First, you need to initialize the GradCache object,
```py
from grad_cache_con_learning import GradCache
from grad_cache_con_learning.losses import ConLoss
...
loss_fn = ConLoss()
gc = GradCache(
  model=model, 
  chunk_size=2, 
  loss_fn=loss_fn,
  loss_type="SelfCon"
)
...
```
Only replace:
```py
...
optimizer.zero_grad()
features = model(images)
f1, f2 = features
features = torch.cat([f.unsqueeze(1) for f in f1] + [f2.unsqueeze(1)], dim=1)
loss = criterion(features, labels)
loss.backward()
optimizer.step()
...
```
with the following:
```py
...
optimizer.zero_grad()
gc(images, labels)
optimizer.step()
...
``` 

