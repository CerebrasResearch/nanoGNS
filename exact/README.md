# Exact Per-Example Gradient Norms for GNS

> Reference implementation of simultaneous per-example norm calculation and
> replication of the experiments found in 
> ["Normalization Layer Per-Example Gradients are Sufficient to Predict Gradient Noise Scale in Transformers"][paper]

## Setup

This repository is a fork of [nanoGPT][] with minimal changes to demonstrate
how to integrate simultaneous per-example gradient norm calculation into an
existing training loop.

Requirements:

- `pytorch`
- `numpy`
- `datasets` (for preparing OpenWebText)
- `tiktoken` (for tokenizing OpenWebText)
- `tqdm` (for progress bars when preparing OpenWebText)

Optional:

- `transformers` (for loading checkpoints)
- `pandas` (for plotting metrics and analyzing GNS)
- `plotille` (for plotting metrics on the command line)
- `wandb` (for logging metrics to Weights & Biases)

## quick start

With per-example gradient norms we can compute the gradient noise scale (GNS)
on anything, even tiny experiments like shakespeare-char. First, I'll go through
how to run that experiment (it'll run on a laptop no problem). Skip to the next
section to see how to run the experiment on OpenWebText from [the paper][paper].

### Shakespeare character-level language model

Prepare the dataset:
```bash
python data/shakespeare_char/prepare.py
```
To train the network with per-example gradient norms, the layer types must be
selected. For example, on CPU, for complete per-example gradient norms:
```bash
python train.py config/train_shakespeare_char.py --device=cpu --linearclass=gns --embeddingclass=gns --lnclass=shim
```
While training you'll see the normal nanoGPT status bar:
```
step 0: train loss 4.2981, val loss 4.2948
iter_num=0: train_lossf=4.2791, dt=160.034s, mfu=0.00%, tokens_per_sec=102
iter_num=10: train_lossf=4.2741, dt=2.147s, mfu=0.00%, tokens_per_sec=7,630
iter_num=20: train_lossf=4.2500, dt=2.101s, mfu=0.17%, tokens_per_sec=7,799
iter_num=30: train_lossf=4.2200, dt=2.425s, mfu=0.17%, tokens_per_sec=6,757
iter_num=40: train_lossf=4.1861, dt=2.350s, mfu=0.17%, tokens_per_sec=6,973
iter_num=50: train_lossf=4.1403, dt=2.518s, mfu=0.17%, tokens_per_sec=6,507
iter_num=60: train_lossf=4.0793, dt=2.232s, mfu=0.17%, tokens_per_sec=7,339
iter_num=70: train_lossf=4.0098, dt=2.352s, mfu=0.17%, tokens_per_sec=6,965
iter_num=80: train_lossf=3.9290, dt=2.691s, mfu=0.16%, tokens_per_sec=6,089
iter_num=90: train_lossf=3.8537, dt=2.748s, mfu=0.16%, tokens_per_sec=5,963
...
```
It doesn't include GNS because GNS is computed offline. To view GNS, run (this
will only plot to the command line if `plotille` is installed):
```bash
python gns-analysis.py out-shakespeare-char --alpha 0.95
```
In addition to the plots, this will print the current GNS, example at iteration
240:
```
...
GNS Analysis Results:
====================
Final GNS: 21.9598
Final G^TG (EMA): 0.9331
Final tr(Σ) (EMA): 20.4898
Average batch size: 64.0
Total tokens processed: 4,096,000
...
```
The GNS for this experiment is starts low and only begins to increase around 120
iterations, which takes about 10 minutes on my laptop. If you have a Mac
M1 series processor:
```bash
python train.py config/train_shakespeare_char.py --device=mps --compile=False --linearclass=gns --embeddingclass=gns --lnclass=shim --device_name=M1_eflops
```
I had to run with `compile=False` but this might be fixed in newer versions.
`--device_name=M1_eflops` is a special flag that tells the script the device to
use for MFU estimation. Other M series processors would have to be added to
`model.py` using [this script to compute the estimated FLOP
ceiling](https://gist.github.com/gaviag-cerebras/5dd1fa407077a3728acc622d33438621).
The above command runs at about 30% MFU for 19k tokens/sec.

### OpenWebText

To prepare OpenWebText:
```bash
python data/openwebtext/prepare.py
```

Unlike the original nanoGPT repository, we wanted to demonstrate how this can
monitor GNS on a single device, so the example code doesn't require DDP. It's
still possible to run DDP using `torchrun` as the original nanoGPT repository
(see below).

The experiment config is set up to run on a single A10 GPU but any single device
that is large enough will work (set the `device_name` to get accurate MFU
estimates though). To run the experiment with GNS collection on all layers:
```bash
python train.py config/train_cgpt_111M_owt.py --linearclass=gns --embeddingclass=gns --lnclass=shim
```
To run the experiment with GNS collection using the fused LayerNorm CUDA kernel:
```bash
python train.py config/train_cgpt_111M_owt.py --lnclass=fused 
```
Both will complete in less than 12 hours but the fused version is about 10%
faster (the same as disabling these layers altogether). On H100 GPUs for
a larger model, we have observed the difference to be more than 40%.

To replicate the batch size schedule experiment from the paper, set
`bs_schedule`. This defaults to a linear batch size schedule. As noted below,
the schedule is written to a file in the output directory so you can modify it
while the experiment is running, it will be loaded and applied on every step.
```bash
python train.py config/train_cgpt_111M_owt.py --bs_schedule=True
```

The notebook "Replicating Figures.ipynb" contains the code to replicate 
figures from the paper. All of the graphs may be reproduced using the withe the
following two experiments, a baseline experiment to gather GNS data:
```bash
python train.py config/train_cgpt_111M_owt.py --linearclass=gns --embeddingclass=gns --lnclass=shim
```
and a batch size schedule for the batch size schdule plots (gathering LayerNorm
GNS for illustration):
```bash
python train.py config/train_cgpt_111M_owt.py --lnclass=fused --out_dir=out-cgpt-openwebtext-bs_schedule --bs_schedule=True
```

### DDP

For an example, if we were running on 4 A10 GPUs, the following should work:
```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_cgpt_111M_owt.py --gradient_accumulation_steps=16 --lnclass=shim --linearclass=gns --embeddingclass=gns
```
The GNS will be logged and may be plotted using `gns-analysis.py`, if "ddp/gns"
is present in the log CSV it will be reported.

## Map of the repository

This map of the repository is a guide to the files and directories. Files are
generally documented in line. The code should also be simple enough to read line
by line.

```
├── train.py                      # main script for training, same interface as nanoGPT
├── buffered.py                   # Layers implementing per-example gradient norm accumulation in buffers
├── model.py                      # Model definition from nanoGPT with optional cosine attention and spectral QKV normalization
├── Replicating Figures.ipynb     # Notebook replicating figures from the paper (GNS and batch size schedule)
├── config                        # Configuration files for training
│   ├── train_cgpt_111M_owt.py    # 111M Cerebras GPT trained on OpenWebText
│   └── train_shakespeare_char.py # Shakespeare character-level language model
├── configurator.py               # Configuration utility from nanoGPT
├── tracker.py                    # Logs metrics to CSV during training, wraps or replaces wandb
├── test_tracker.py               # Tests for tracker.py
├── plot.py                       # script to plot logged metrics on the command line using plotille
├── gns-analysis.py               # script to plot GNS from logged metrics using plotille
├── csv_tools.py                  # Collection of csv utilities used in gns-analysis.py and plot.py
├── data
│   └── openwebtext               # OpenWebText dataset directory from nanoGPT
│       ├── prepare.py
│       └── readme.md
├── ddpgns.py                     # Utility for computing GNS during DDP training from crowsonkb
├── fused_gns.py                  # Fused CUDA kernel layer definition for per-example gradient norms
├── gnstracking.py                # GNS tracking bookkeeping utility
├── layer_config.py               # Dynamic layer class configuration utility
├── normgnorm                     # Fused CUDA kernel for per-example gradient norms
│   ├── __init__.py               # torch.autograd.Function definition using csrc/normgnorm_cuda.cpp and csrc/normgnorm_cuda.cu
│   ├── csrc
│   │   ├── normgnorm_cuda.cpp
│   │   └── normgnorm_cuda.cu
│   └── tst.py                    # Test script for normgnorm
└── out-cgpt-openwebtext          # example output directory (nanoGPT style)
    ├── config.json               # configuration options that were passed
    ├── gns_analysis.csv          # gns analysis generated by gns-analysis.py
    ├── grad_accum_schedule.txt   # gradient accumulation schedule for batch size scheduling
    └── log.csv                   # metrics that were logged using tracker.py
```

## nanoGPT diff

To illustrate the changes required to integrate simultaneous per-example
gradient norms, here is a diff between the original `train.py` from nanoGPT and
the modified version in this repository. Splitting it up for annotation, the
first difference is the imports required, we need the layers from `buffered.py`
and `fused_gns.py`. The model import is also removed because it must be placed
inside the context manager from `layer_config.py`:

```diff
@@ -27,7 +27,13 @@ import torch
 from torch.nn.parallel import DistributedDataParallel as DDP
 from torch.distributed import init_process_group, destroy_process_group
 
-from model import GPTConfig, GPT
+import layer_config as lc
+from buffered import (PEGradNormShimLinear, PEGradNormShimEmbedding,
+                      PEGradNormSeparatedLayerNorm, PEGradNormLinear,
+                      PEGradNormEmbedding, zero_sqgradnorm_buffers)
+from fused_gns import PEGradNormFusedLayerNorm
+import gnstracking
+from tracker import LogWrapper
```

Some configuration changes are required to select the layer type required. The
remaining changes are for batch size scheduling. For example, `bs_schedule` and
the changes made to denominate training in tokens instead of iterations:

```diff
 # -----------------------------------------------------------------------------
 # default config values designed to train a gpt2 (124M) on OpenWebText
@@ -54,23 +60,33 @@ n_head = 12
 n_embd = 768
 dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
 bias = False # do we use bias inside LayerNorm and Linear layers?
+linearclass = 'nn' # gns:PEGradNormLinear or shim:PEGradNormShimLinear or nn:nn.Module
+embeddingclass = 'nn' # gns:PEGradNormEmbedding or shim:PEGradNormShimEmbedding or nn:nn.Embedding
+lnclass = 'nn' # shim:PEGradNormSeparatedLayerNorm or nn:nn.LayerNorm or fused:PEGradNormFusedLayerNorm
+spectral_c_attn = [] # block indexes to use spectralnorm on QKV projection
+cos_attn = [] # block indexes to use cosine attention
 # adamw optimizer
 learning_rate = 6e-4 # max learning rate
 max_iters = 600000 # total number of training iterations
+max_tokens = 15 * 10**12 # total number of tokens to process
 weight_decay = 1e-1
 beta1 = 0.9
 beta2 = 0.95
 grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
+# batch size schedule
+bs_schedule = False # if True, use a linear schedule to final gradient accumulation steps
 # learning rate decay settings
 decay_lr = True # whether to decay the learning rate
-warmup_iters = 2000 # how many steps to warm up for
-lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
+warmup_tokens = 4000 * 12 * 5 * 8 * 1024 # 4000 steps for 12 * 5 * 8 batch size and 1024 block size
+lr_decay_tokens = 60000 * 12 * 5 * 8 * 1024 # 60000 steps for 12 * 5 * 8 batch size and 1024 block size
 min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
 # DDP settings
 backend = 'nccl' # 'nccl', 'gloo', etc.
 # system
 device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
+device_name = 'A100' # 'A100', 'A10' to use spec sheet FLOPs for MFU, or '*_eflops' for estimated FLOP ceilings
 dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
+fp32_attention_layers=[] # list of layer indices to run in float32
 compile = True # use PyTorch 2.0 to compile the model to be faster
 # -----------------------------------------------------------------------------
 config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
```

The model initialization change results in a big diff but what's really
happening here is we're adding the `lc.set_contextual_config` context manager
so we can control which layer classes are used. The `Linear`, `Embedding`, and
`LayerNorm` classes are selected based on the configuration.

Afterwards a buffer is added to the model to register the number of tokens
processed. This could be stored separately, but it's convenient to have it
stored inside the model and helps with synchronization in DDP training.

```diff
@@ -144,54 +160,91 @@ if os.path.exists(meta_path):
     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
 
 # model init
-model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
-                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
-if init_from == 'scratch':
-    # init a new model from scratch
-    print("Initializing a new model from scratch")
-    # determine the vocab size we'll use for from-scratch training
-    if meta_vocab_size is None:
-        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
-    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
-    gptconf = GPTConfig(**model_args)
-    model = GPT(gptconf)
-elif init_from == 'resume':
-    print(f"Resuming training from {out_dir}")
-    # resume training from a checkpoint.
-    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
-    checkpoint = torch.load(ckpt_path, map_location=device)
-    checkpoint_model_args = checkpoint['model_args']
-    # force these config attributes to be equal otherwise we can't even resume training
-    # the rest of the attributes (e.g. dropout) can stay as desired from command line
-    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
-        model_args[k] = checkpoint_model_args[k]
-    # create the model
-    gptconf = GPTConfig(**model_args)
-    model = GPT(gptconf)
-    state_dict = checkpoint['model']
-    # fix the keys of the state dictionary :(
-    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
-    unwanted_prefix = '_orig_mod.'
-    for k,v in list(state_dict.items()):
-        if k.startswith(unwanted_prefix):
-            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
-    model.load_state_dict(state_dict)
-    iter_num = checkpoint['iter_num']
-    best_val_loss = checkpoint['best_val_loss']
-elif init_from.startswith('gpt2'):
-    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
-    # initialize from OpenAI GPT-2 weights
-    override_args = dict(dropout=dropout)
-    model = GPT.from_pretrained(init_from, override_args)
-    # read off the created config params, so we can store them into checkpoint correctly
-    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
-        model_args[k] = getattr(model.config, k)
+model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
+                  block_size=block_size, bias=bias, vocab_size=None,
+                  dropout=dropout, device_name=device_name,
+                  spectral_c_attn=spectral_c_attn,
+                  cos_attn=cos_attn) # start with model_args from command line
+# parsing the layer classes from the config
+Linear = {'gns': PEGradNormLinear, 'nn': torch.nn.Linear,
+          'shim': PEGradNormShimLinear}[linearclass]
+Embedding = {'gns': PEGradNormEmbedding, 'nn': torch.nn.Embedding,
+             'shim': PEGradNormShimEmbedding}[embeddingclass]
+LayerNorm = {'shim': PEGradNormSeparatedLayerNorm,
+             'nn': torch.nn.LayerNorm,
+             'fused': PEGradNormFusedLayerNorm}[lnclass]
+with lc.set_contextual_config(Linear=Linear, Embedding=Embedding,
+                              LayerNorm=LayerNorm):
+    # import must happen inside the context manager so the config is set
+    from model import GPTConfig, GPT
+    if init_from == 'scratch':
+        # init a new model from scratch
+        print("Initializing a new model from scratch")
+        # determine the vocab size we'll use for from-scratch training
+        if meta_vocab_size is None:
+            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
+        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
+        gptconf = GPTConfig(**model_args)
+        model = GPT(gptconf)
+    elif init_from == 'resume':
+        print(f"Resuming training from {out_dir}")
+        # resume training from a checkpoint.
+        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
+        checkpoint = torch.load(ckpt_path, map_location=device)
+        checkpoint_model_args = checkpoint['model_args']
+        # force these config attributes to be equal otherwise we can't even resume training
+        # the rest of the attributes (e.g. dropout) can stay as desired from command line
+        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
+            model_args[k] = checkpoint_model_args[k]
+        # create the model
+        gptconf = GPTConfig(**model_args)
+        model = GPT(gptconf)
+        state_dict = checkpoint['model']
+        # fix the keys of the state dictionary :(
+        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
+        unwanted_prefix = '_orig_mod.'
+        for k,v in list(state_dict.items()):
+            if k.startswith(unwanted_prefix):
+                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
+        model.load_state_dict(state_dict)
+        iter_num = checkpoint['iter_num']
+        best_val_loss = checkpoint['best_val_loss']
+    elif init_from.startswith('gpt2'):
+        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
+        # initialize from OpenAI GPT-2 weights
+        override_args = dict(dropout=dropout)
+        model = GPT.from_pretrained(init_from, override_args)
+        # read off the created config params, so we can store them into checkpoint correctly
+        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
+            model_args[k] = getattr(model.config, k)
+
+# add buffer to the model for the number of tokens processed
+model.register_buffer('tokens_processed', torch.tensor(0, dtype=torch.int64))
+
 # crop down the model block size if desired, using model surgery
 if block_size < model.config.block_size:
     model.crop_block_size(block_size)
     model_args['block_size'] = block_size # so that the checkpoint will have the right value
 model.to(device)
```

Immediately afterwards we add an option for numerical stability without using
the cosine attention or spectral normalization options. This is mostly for
sanity checking because we found in testing that this would not work with
`torch.compile`. However, it could be useful in future if later `torch.compile` 
versions support this feature. It ensures that the specified layers are run in
`float32` regardless of the `dtype` set.

```diff
+# numerical stability monkey patch hack
+# in PyTorch 2.4.0 this breaks compile, so not a preferred option
+# see: https://gist.github.com/gaviag-cerebras/b77aef9de29e859a5e999a582d57f6a2
+# approx 2% reduction in MFU with one block using fp32
+if fp32_attention_layers:
+    from functools import wraps
+    # make context manager for float32
+    def use_context(method):
+        @wraps(method)
+        def wrapper(self, x):
+            with self.ctx: # first argument is self
+                return method(x)
+        return wrapper
+    for layer_idx in fp32_attention_layers:
+        attn = model.transformer.h[layer_idx].attn
+        attn.ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float32)
+        attn.forward = use_context(attn.forward).__get__(attn)
+
 # initialize a GradScaler. If enabled=False scaler is a no-op
 scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

The next change is to add the GNS tracking. These classes are responsible for
keeping track of the buffers that store the per-example gradient norms in each
layer. `gns_tracker` stores references to these buffers and gathers them during
training when `.step` is called, which allows `accumulate_gns` to log the GNS
statistics. These are later passed to the `tracker` for logging.

When disabled, the `DoNothing` class does nothing. Specifically, it doesn't log
anything and doesn't pass anything to the tracker.

```diff 
@@ -201,6 +254,22 @@ if init_from == 'resume':
     optimizer.load_state_dict(checkpoint['optimizer'])
 checkpoint = None # free up memory
 
+# gns tracking
+class DoNothing: # class that does nothing, for when we don't want to track GNS
+    def __getattr__(self, name):
+        return lambda *args, **kwargs: {}
+tracked = linearclass != 'nn' or embeddingclass != 'nn' or lnclass != 'nn'
+if tracked and master_process:
+    accumulate_gns = gnstracking.AccumulateMeasurements(prefix="train/gns/") # for logging
+    gns_tracker = gnstracking.MeasurementTracker.from_model(
+        model,
+        callback=accumulate_gns,
+        scaler=scaler if dtype == 'float16' else None,
+    )
+else:
+    gns_tracker = DoNothing()
+    accumulate_gns = DoNothing()
+


 # compile the model
 if compile:
     print("compiling the model... (takes a ~minute)")
```

If we're training with DDP we can also track the GNS independently via the `ddpgns`
module. This is a slightly adapted version of [crowsonkb's DDP GNS tracking code](https://github.com/crowsonkb/k-diffusion/blob/21d12c91ad4550e8fcf3308ff9fe7116b3f19a08/k_diffusion/gns.py).

```diff
@@ -210,6 +279,14 @@ if compile:
 # wrap model into DDP container
 if ddp:
     model = DDP(model, device_ids=[ddp_local_rank])
+# gns tracking with DDP
+ddp_gns_enabled = tracked and ddp
+if ddp_gns_enabled:
+    ddp_gns_stats_hook = ddpgns.DDPGradientStatsHook(model)
+    ddp_gns_stats = ddpgns.GradientNoiseScale()
+else:
+    ddp_gns_stats = DoNothing()
+
 
 # helps estimate an arbitrarily accurate loss over either split using many batches
 @torch.no_grad()
```

This change ensures that we don't pass a torch tensor to the tracker, which
nanoGPT gets away with not doing because `wandb` converts it to a float itself.

```diff
@@ -223,28 +300,56 @@ def estimate_loss():
             with ctx:
                 logits, loss = model(X, Y)
             losses[k] = loss.item()
-        out[split] = losses.mean()
+        out[split] = losses.mean().item()
     model.train()
     return out
```

This code implements the batch size schedule for replicating the batch size
schedule experiment in the paper. As noted in the comments, it writes the batch
size schedule to a file in the output directory so you can modify the schedule
while the experiment is running by editing the file.

```diff
-# learning rate decay scheduler (cosine with warmup)
-def get_lr(it):
-    # 1) linear warmup for warmup_iters steps
-    if it < warmup_iters:
-        return learning_rate * it / warmup_iters
-    # 2) if it > lr_decay_iters, return min learning rate
-    if it > lr_decay_iters:
+# batch size schedule
+# generate a piecewise linear schedule for gradient accumulation steps
+# written to disk so you can change it on the fly if you want
+if bs_schedule and master_process:
+    n_points = 10
+    grad_accum_steps = np.linspace(2, gradient_accumulation_steps, n_points, dtype=np.int64)
+    grad_accum_tokens = np.interp(grad_accum_steps,
+                                  [grad_accum_steps[0], grad_accum_steps[-1]],
+                                  [0, max_tokens]).astype(np.int64) # interpolate to prevent aliasing
+    with open(os.path.join(out_dir, 'grad_accum_schedule.txt'), 'w') as f:
+        tokformat = lambda t: format(t, ',').replace(',', '_') # easier to read
+        f.write("\n".join(f"{tokformat(t)}, {s:.0f}" for t,s in zip(grad_accum_tokens, grad_accum_steps)))
+# function that interpolates this schedule
+def get_grad_accum_steps(tokens):
+    if bs_schedule:
+        with open(os.path.join(out_dir, 'grad_accum_schedule.txt'), 'r') as f:
+            grad_accum_tokens, grad_accum_steps = zip(*[map(float, line.strip().split(', ')) for line in f])
+        ga_steps = np.interp(tokens, grad_accum_tokens, grad_accum_steps)
+        ga_steps = math.ceil(ga_steps / ddp_world_size) # scale down to per-process
+        return ga_steps
+    else:
+        return gradient_accumulation_steps // ddp_world_size
+
```

Next we need to modify the learning rate schedule in two ways:

1. We need it to be denominated in tokens instead of iterations because
  otherwise we'd have to carefully set the schedule to match the batch size
  schedule.
2. We need to change the cosine decay to a linear decay to match the
   Cerebras-GPT training recipe.

```diff
+# learning rate decay scheduler (linear with warmup)
+def get_lr(tokens):
+    lr = learning_rate
+    # 1) linear warmup for warmup_tokens steps
+    if tokens < warmup_tokens:
+        return lr * tokens / warmup_tokens
+    # 2) if tokens > lr_decay_tokens, return min learning rate
+    if tokens > lr_decay_tokens:
         return min_lr
     # 3) in between, use linear decay down to min learning rate
-    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
-    assert 0 <= decay_ratio <= 1
-    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
-    return min_lr + coeff * (learning_rate - min_lr)
+    decay_ratio = (tokens - warmup_tokens) / (lr_decay_tokens - warmup_tokens)
+    coeff = 1.0 - decay_ratio
+    return min_lr + coeff * (lr - min_lr)
```

We wanted a way to make sure we always log the GNS metrics without having to
rely on `wandb` so we added a `tracker` object that wraps the `wandb.log` or
operates independently. It's necessary because the GNS is best computed offline
from measured statistics so the EMA parameters can be adjusted (this can be done
using `gns-analysis.py`).

```diff
 # logging
-if wandb_log and master_process:
-    import wandb
-    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
+log = None
+if master_process:
+    if wandb_log:
+        import wandb
+        run = wandb.init(project=wandb_project, name=wandb_run_name, config=config)
+        def log(msg):
+            run.log({k:v for k,v in msg.items()})
+    tracker = LogWrapper(log, config=config, out_dir=out_dir) # wraps log function and writes csv
 
 # training loop
 X, Y = get_batch('train') # fetch the very first batch
```

We had to add a `final_iter` variable so we could make sure we always got
a reading of the validation loss on the final iteration.

```diff
@@ -252,25 +357,24 @@ t0 = time.time()
 local_iter_num = 0 # number of iterations in the lifetime of this process
 raw_model = model.module if ddp else model # unwrap DDP container if needed
 running_mfu = -1.0
+final_iter = False if not eval_only else True
 while True:
```

The following changes to the training loop make use of the changes to the
schedules described above. We can also see how the `tracker` has a simplified
interface. The variables that are not passed to `tracker.log` here are going
to be logged later in the training loop so there's no need to pass them.

```diff
     # determine and set the learning rate for this iteration
-    lr = get_lr(iter_num) if decay_lr else learning_rate
+    lr = get_lr(raw_model.tokens_processed.item()) if decay_lr else learning_rate
     for param_group in optimizer.param_groups:
         param_group['lr'] = lr
+    ga_steps = get_grad_accum_steps(raw_model.tokens_processed.item())
+    tokens_per_iter = ga_steps * ddp_world_size * batch_size * block_size
 
     # evaluate the loss on train/val sets and write checkpoints
-    if iter_num % eval_interval == 0 and master_process:
+    if (final_iter or (iter_num % eval_interval == 0)) and master_process:
         losses = estimate_loss()
         print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
-        if wandb_log:
-            wandb.log({
-                "iter": iter_num,
-                "train/loss": losses['train'],
-                "val/loss": losses['val'],
-                "lr": lr,
-                "mfu": running_mfu*100, # convert to percentage
-            })
+        tracker.log({
+            "train/loss": losses['train'],
+            "val/loss": losses['val'],
+        })
         if losses['val'] < best_val_loss or always_save_checkpoint:
             best_val_loss = losses['val']
             if iter_num > 0:
@@ -284,29 +388,45 @@ while True:
                 }
                 print(f"saving checkpoint to {out_dir}")
                 torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
-    if iter_num == 0 and eval_only:
+    if iter_num == 0 and (eval_only or final_iter):
         break
```

We've defined a new variable `ga_steps` that is the dynamic gradient
accumulation steps used for batch size scheduling.

> [!IMPORTANT]
> `gns_tracker.step` is called immediately after all of the gradient
> accumulation microbatches and after the final backward pass. This gathers
> the per-example gradient norms stored in the buffers and triggers any callbacks
> that are registered. It is important that this happens before gradient clipping
> because that would affect the full batch gradient norm logged.
>
> Immediately after that we zero the buffers manually. This is also necessary
> otherwise on the next step the buffers would already store a per-example gradient
> norm. This is performed separately because during DDP training the non-master
> processes have a `DoNothing` tracker that won't zero the buffers as they're
> read (which is what `gns_tracker` normally does).
>
> This is also the last time we need to do anything with `gns_tracker`.

Finally, we need to keep track of the number of tokens processed so we add to
the tokens processed buffer after we've processed some tokens in the forward and
backward passes.

```diff
     # forward backward update, with optional gradient accumulation to simulate larger batch size
     # and using the GradScaler if data type is float16
-    for micro_step in range(gradient_accumulation_steps):
+    for micro_step in range(ga_steps):
         if ddp:
             # in DDP training we only need to sync gradients at the last micro step.
             # the official way to do this is with model.no_sync() context manager, but
             # I really dislike that this bloats the code and forces us to repeat code
             # looking at the source of that context manager, it just toggles this variable
-            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
+            model.require_backward_grad_sync = (micro_step == ga_steps - 1)
         with ctx:
             logits, loss = model(X, Y)
-            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
+            loss = loss / ga_steps # scale the loss to account for gradient accumulation
         # immediately async prefetch next batch while model is doing the forward pass on the GPU
         X, Y = get_batch('train')
         # backward pass, with gradient scaling if training in fp16
         scaler.scale(loss).backward()
+    gns_tracker.step(batch_size=ga_steps * batch_size * ddp_world_size) # track gradient noise scaling
+    if tracked:
+        zero_sqgradnorm_buffers(raw_model)
+    if master_process: # track number of tokens processed
+        raw_model.tokens_processed.add_(tokens_per_iter)
+
```

`gns_tracker` logs the norm of the gradients but it's useful for us to log the
gradient norm regardless of whether that's enabled, so we're also going to log
the gradient norm manually here.

After that is the logic required to call the DDP GNS tracking code. This could
have been wrapped but we kept it the same as the original codebase.

```diff
     # clip the gradient
     if grad_clip != 0.0:
         scaler.unscale_(optimizer)
-        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
+        parameters = [param for group in optimizer.param_groups for param in group['params']]
+        grad_norm = math.sqrt(sum(p.grad.norm().item()**2 for p in parameters))
+        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
+    if ddp_gns_enabled:
+        sq_norm_small_batch = ddp_gns_stats_hook.get_stats()
+        sq_norm_large_batch = grad_norm**2
+        big_batch_size = X.shape[0] * ddp_world_size * ga_steps
+        small_batch_size = X.shape[0] * ga_steps
+        ddp_gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, small_batch_size, big_batch_size)
+        ddp_gns_measurement = gnstracking.Measurement(sq_norm_large_batch**0.5, sq_norm_small_batch**0.5, big_batch_size, small_batch_size)
+
```

> [!WARNING]
> `tracker.py` must call `tracker.step()` in order to log anything to the CSV file.
> This is a change from `wandb` so it might trip people up. I decided this
> would be more reliable and explicit (the same thing can be achieved with
> `commit=False` in `wandb`).
>
> Another less dangerous difference is we use `tracker.print` as a shorthand to print
> a status to the command line. `tracker.print` is effectively an f-string that
> allows the user to print whatever is currently cached (ie will be logged when
> `.step` is called next). It also uses a format string that's not real:
> `{train_lossf=:.4f}` will be printed as `train/lossf=<value to 4 decimal places>` (underscore is mapped to /). 
> This isn't necessary but I think it's neat.

Methods like `msg` and `get_msg` pass dictionaries to the tracker. We log
`accumulate_gns.get_msg()` on every iteration because we're logging the
per-example norms on every iteration, so that must be outside the `if` block.

```diff
     # step the optimizer and scaler if training in fp16
     scaler.step(optimizer)
     scaler.update()
@@ -320,17 +440,44 @@ while True:
     if iter_num % log_interval == 0 and master_process:
         # get loss as float. note: this is a CPU-GPU sync point
         # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
-        lossf = loss.item() * gradient_accumulation_steps
+        lossf = loss.item() * ga_steps
         if local_iter_num >= 5: # let the training loop settle a bit
-            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
+            mfu = raw_model.estimate_mfu(batch_size * ga_steps, dt)
             running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
-        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
+        tracker.log({
+            "dt": dt,
+            "tokens_per_sec": tokens_per_iter / dt,
+            "mfu": 100*running_mfu if iter_num > 10 else 0.0,
+            "lr": lr,
+            "grad_norm": grad_norm,
+            "gradient_accumulation_steps": ga_steps,
+            "train/lossf": lossf,
+        })
+        tracker.print(f"{iter_num=}: "+"{train_lossf=:.4f}, dt={dt:.3f}s, {mfu=:.2f}%, {tokens_per_sec=:,.0f}")
+        if ddp_gns_enabled:
+            tracker.log(ddp_gns_measurement.msg(prefix="ddp/"))
+            tracker.log(ddp_gns_stats.get_msg(prefix="ddp/"))
+    # this actually writes the logs to csv and calls logf (wandb or aim)
+    tracker.log({
+        **accumulate_gns.get_msg(),
+    })
+    if tracker.log_dict: # if we're going to log something, log the index vars
+        tracker.log({
+            "iter_num": iter_num,
+            "tokens_processed": raw_model.tokens_processed.item(),
+        })
+    tracker.step()
+
     iter_num += 1
     local_iter_num += 1
```

Finally, as noted above we want to have a final validation loss iteration so we
modify the termination conditions to include `final_iter` which will trigger the
eval loop and then quit.

```diff
     # termination conditions
-    if iter_num > max_iters:
+    if final_iter:
         break
-
+    if iter_num > max_iters:
+        final_iter = True
+    if max_tokens is not None:
+        if raw_model.tokens_processed.item() > max_tokens:
+            final_iter = True
 if ddp:
     destroy_process_group()
```

## Bibtex

```bibtex
TODO (NeurIPS page doesn't exist yet)
```

[nanogpt]: https://github.com/karpathy/nanoGPT
[paper]: https://neurips.cc/virtual/2024/poster/95128
