
# nanoGNS

> Reference implementation of efficient per-example GNS and approximate 
> per-example GNS on nanoGPT for replicating 
> ["Efficient and Approximate Per-Example Gradient Norms for Gradient Noise Scale"][sogns].

Tested on CPU and local DDP GPU.

Additional MNIST implementation and experiment can be found in 
[this gist](https://gist.github.com/gaviag-cerebras/aa8050a2b4a2f327c83bc7b21f9e8b89).

## Install

The install script will symlink `hook.py` and `gns_utils.py` into the
`examples/nanoGPT` subdirectory.

```
chmod +x install.sh
./install.sh
```

## Running examples/nanoGPT

Example experiment that runs on CPU:

```
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=32 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 --gns_type=sogns
```

This will run with Scaled Output Gradient Noise Scale (see [our paper][sogns]).

The code also supports exact per-example Gradient Noise scale (but it is
inefficient):

```
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=32 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0 --gns_type=exact
```

The original nanoGPT repository may be found [here](https://github.com/karpathy/nanoGPT).

### GPU DDP

The following 4 GPU config has been tested, but it should work the same on other
DDP configs:

```
torchrun --standalone --nproc_per_node=4 train.py config/train_shakespeare_char_4gpu.py --gns_type=sogns --dtype=float16
```

## Code details

The complete diff in `train.py` required to add GNS logging to nanoGPT is:

```diff
diff --git a/train.py b/train.py
index a482ab7..1e22585 100644
--- a/train.py
+++ b/train.py
@@ -28,6 +28,11 @@ from torch.nn.parallel import DistributedDataParallel as DDP
 from torch.distributed import init_process_group, destroy_process_group
 
 from model import GPTConfig, GPT
+from hook import (add_hooks_to_model, add_sogns_hooks,
+                  add_exact_hooks,  gather_hook_results)
+
+import gns_utils
+
 
 # -----------------------------------------------------------------------------
 # default config values designed to train a gpt2 (124M) on OpenWebText
@@ -39,6 +44,7 @@ eval_iters = 200
 eval_only = False # if True, script exits right after the first eval
 always_save_checkpoint = True # if True, always save a checkpoint after each eval
 init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
+gns_type = 'sogns' # 'sogns' or 'exact'
 # wandb logging
 wandb_log = False # disabled by default
 wandb_project = 'owt'
@@ -152,6 +158,11 @@ if init_from == 'scratch':
     model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
     gptconf = GPTConfig(**model_args)
     model = GPT(gptconf)
+    if gns_type == 'sogns':
+        add_hooks_to_model(model, add_sogns_hooks)
+    elif gns_type == 'exact':
+        add_hooks_to_model(model, add_exact_hooks)
+    gns_ema = gns_utils.EMA(beta=0.9)
 elif init_from == 'resume':
     print(f"Resuming training from {out_dir}")
     # resume training from a checkpoint.
@@ -259,7 +270,8 @@ while True:
     # evaluate the loss on train/val sets and write checkpoints
     if iter_num % eval_interval == 0 and master_process:
         losses = estimate_loss()
-        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
+        gns = gns_ema.get_gns()
+        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, gns {gns:.2f}")
         if wandb_log:
             wandb.log({
                 "iter": iter_num,
@@ -300,6 +312,8 @@ while True:
         X, Y = get_batch('train')
         # backward pass, with gradient scaling if training in fp16
         scaler.scale(loss).backward()
+        approx_gns_results = gather_hook_results(model)
+        gns_ema.update(*gns_utils.gnsify(approx_gns_results, batch_size))
     # clip the gradient
     if grad_clip != 0.0:
         scaler.unscale_(optimizer)
@@ -321,7 +335,9 @@ while True:
         if local_iter_num >= 5: # let the training loop settle a bit
             mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
             running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
-        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
+        gns = gns_ema.get_gns()
+        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, gns {gns:.2f}")
+
     iter_num += 1
     local_iter_num += 1
 
```

### `gns_utils.py`

This contains tools to deal with the GNS statistics but does not compute the
per-example gradient norms:

* `GradNorm`: dataclass for storing and annotating gradient norms
* `mean_loss_scale`: trivial function for equivalent scaling of small and large
    batch gradient norms
* `EMA`: an exponential moving average
* `gnsify`: utility function to convert a dictionary of scalar gradient norms
    into `GradNorm` objects

### `hook.py`

Contains functionality for accessing gradients using backward hooks and
computing per-example gradient norms:


* `add_sogns_hooks`: adds hooks required for computing scaled output gradient 
    noise scale 
* `add_exact_hooks`: adds hooks required for computing per-example parameter
    gradient noise scale
* `add_hooks_to_model`: adds hooks to all linear layers in a model
* `HookResult`: trivial dataclass for storing per-example and full minibatch
    gradient norms
* `gather_hook_results`: extracts the results of the processing performed by the
    hooks that gets stored in each Linear module

## Bibtex

```bibtex
@inproceedings{
gray2023efficient,
title={Efficient and Approximate Per-Example Gradient Norms for Gradient Noise Scale},
author={Gavia Gray and Anshul Samar and Joel Hestness},
booktitle={Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@NeurIPS 2023)},
year={2023},
url={https://openreview.net/forum?id=xINTMAvPQA}
}
```

[sogns]: https://openreview.net/forum?id=xINTMAvPQA
