# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
# out_dir = 'out-multiplication'

wandb_log = False
wandb_project = 'test'
wandb_run_name='multiplication'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 64
block_size = 4096
gradient_accumulation_steps = 1
always_save_checkpoint = False

# this makes total number of tokens be 300B
max_iters = 20000
lr_decay_iters = 3000
warmup_iters = 200

# eval stuff
eval_interval = 200
eval_iters = 50
log_interval = 50

# weight decay
weight_decay = 1e-1

learning_rate = 2e-5
min_lr = 2e-6

