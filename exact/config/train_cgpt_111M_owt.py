out_dir = 'out-cgpt-openwebtext'
dataset = 'openwebtext'
wandb_run_name = 'cgpt_111M'
# this is set for A10 GPUs, modify for other GPUs
#device_name = 'A10_eflops' # empirical FLOP ceiling for A10 GPU
device_name = 'A10' # datasheet FLOP ceiling for A10 GPU
batch_size = 8
block_size = 2048
gradient_accumulation_steps = 15
learning_rate = 1.17e-3
min_lr = learning_rate / 10
max_tokens = 2_219_950_080 # 9033 * 8 * 15 * 2048
lr_decay_tokens = 2_219_950_080 # 9033 * 8 * 15 * 2048
warmup_tokens = 374_784_000 # 1525 * 8 * 15 * 2048
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
init_from = 'scratch'
# model
n_layer = 10
n_head = 12
n_embd = 768
dropout = 0.0
bias = True
cos_attn = [1] # necessary for numerical stability (See Appendix C.2), results are unchanged
# eval stuff
eval_interval = 1000
log_interval = 10
# weight decay
weight_decay = 1e-1
