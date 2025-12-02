"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from dataset import VerticalMultiplicationDataset, decode, decode_to_result, MultiplicationHeatMapDataset
from model import GPTConfig, GPT, PositionEmbedding

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 640 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 3 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
torch.serialization.add_safe_globals([PositionEmbedding])
# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

from torch.utils.data import DataLoader


canvas_width = 20
canvas_height = 30
max_num_length = 6
run_times = 4
total_size = max_num_length * max_num_length
batch_size = total_size * 4

def get_correct_ratio(run_times:int, batch_size:int, num_len_1:int, num_len_2:int):
    
    correct_times = 0
    total_times = run_times * batch_size
    

    # run generation
    with torch.no_grad():
        with ctx:
            for i in range(run_times):
                data = next(dataloader_iter)
                x:torch.Tensor = data["inputs"]
                label:torch.Tensor = data["labels"]
                x = x.to(device)
                y:torch.Tensor = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                for b in range(batch_size):
                    ref = decode_to_result(label[b].tolist())
                    answer = decode_to_result(y[b].tolist())
                    print(f"runs[{i}][{b}]: ref {ref}, answer {answer}")
                    if ref == answer:
                        correct_times += 1
    return correct_times / total_times



import numpy as np
correct_nums = np.zeros([max_num_length, max_num_length], dtype=np.float32)
total_nums = np.zeros([max_num_length, max_num_length], dtype=np.float32)

# for i in range(1, max_num_length + 1):
#     for j in range(1, max_num_length + 1):
#         print(f"correct ratio of {i}, {j}")
#         correct_ratios[i-1][j-1] = get_correct_ratio(1, 16, i, j)

dataset = MultiplicationHeatMapDataset(
    length=10000000000, 
    seed=36, 
    width=canvas_width, 
    height=canvas_height, 
    min_num_len_1=1,
    max_num_len_1=max_num_length,
    min_num_len_2=1,
    max_num_len_2=max_num_length,
    with_add=False,
    for_test=True
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,# 每批次样本数
    shuffle=False,       # 是否打乱数据
    num_workers=1,       # 加载数据的线程数
    pin_memory=True,     # 若使用GPU可提升速度
)

dataloader_iter = iter(dataloader)

index = 0

with torch.no_grad():
    with ctx:        
        for i in range(run_times):
            print("run time:", i)
            data = next(dataloader_iter)
            x:torch.Tensor = data["inputs"]
            label:torch.Tensor = data["labels"]
            x = x.to(device)
            y:torch.Tensor = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            for b in range(batch_size):
                len_a = index // max_num_length
                len_b = index % max_num_length
                ref = decode_to_result(label[b].tolist())
                answer = decode_to_result(y[b].tolist())
                print(f"a{len_a} b{len_b}: ref {ref}, answer {answer}")
                index += 1 
                index = index % total_size
                total_nums[len_a][len_b] += 1 
                if ref == answer:
                    correct_nums[len_a][len_b] += 1
            
            
            
            
            
correct_ratios = correct_nums / total_nums
            # for j in range(max_num_length):
            #     start = j * each_point_num
            #     correct_times = 
            #     for k in range(0, each_point_num):
            #         index = start + k
            #         ref = decode_to_result(label[index].tolist())
            #         answer = decode_to_result(y[index].tolist())
            #         print(f"runs[{index}]: ref {ref}, answer {answer}")
            #         if ref == answer:
            #             correct_times += 1
    

import matplotlib.pyplot as plt
# plt.figure(figsize=(6, 5))
# im = plt.imshow(correct_ratios, cmap='seismic', vmin=0, vmax=1, origin='lower')
# plt.colorbar(im, label='Correct Ratio')

# plt.xlabel('Number length j')
# plt.ylabel('Number length i')
# plt.title('Correct Ratio Heatmap (Blue=1, Red=0)')

# plt.tight_layout()

# # 保存为高分辨率PNG
# plt.savefig("correct_ratio_heatmap.png", dpi=300)
# plt.close()

plt.figure(figsize=(6, 5))
im = plt.imshow(
    correct_ratios,
    cmap='seismic_r',   # 蓝(高,正确) → 白 → 红(低,错误)
    vmin=0,
    vmax=1,
    origin='upper'      # 左上角为原点
)

plt.colorbar(im, label='Correct Ratio')

max_num_length = correct_ratios.shape[0]
plt.xticks(np.arange(max_num_length), np.arange(1, max_num_length + 1))
plt.yticks(np.arange(max_num_length), np.arange(1, max_num_length + 1))

plt.xlabel('Number length j')
plt.ylabel('Number length i')
plt.title('Correct Ratio Heatmap')

plt.tight_layout()
plt.savefig("correct_ratio_heatmap_pid.png", dpi=300)
plt.close()






