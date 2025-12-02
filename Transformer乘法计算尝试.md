# Transformer乘法计算尝试

## NanoGPT(GPT2)

nanoGPT的项目连接位https://github.com/karpathy/nanoGPT
实验代码位于https://github.com/birdswimming/NanoGPT-multi

使用最简单的GPT2模型，使用固定大小的画布（如20*20）作为训练的输入，将一个完整的乘法竖式计算过程放置在画布中，且对数据进行了翻转，使得token预测的顺序与人类计算的顺序一致。生成的数据样例如下，竖式可以在画布内随机位置出现。

```shell
____________________#
____________________#
____________________#
____________________#
____________________#
____________________#
____________________#
____________________#
____________________#
____________________#
____________________#
____________________#
____________321_____#
____________654_*___#
____________837_____#
_____________516____#
______________294___#
____________88065___#
____________________#
____________________#
```

使用的GPT规模如下

| n_layer | n_head | n_embd | 参数总量 |
| ------- | ------ | ------ | -------- |
| 12      | 12     | 768    | 120+M    |

该问题的关键在于位置编码，需要使用合适的位置编码用以表示画布每一行之间的对齐关系，尝试有如下的位置编码。

### ABS_LEARNED

```python
wpe = nn.Embedding(config.block_size, config.n_embd)
pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
in_emb = pos_emb + tok_emb
```

使用一个最大上下文长度（block_size）*词向量维度（n_embd）的权重矩阵为每个位置保存一个位置向量。值得注意的是，在Deepseek-OCR中，ViT部分使用的也是这种位置编码。

在5位×5位的乘法数据集下训练后效果不佳，基本没有泛化能力。在数据中额外添加加法部分的计算过程后能够在与训练集相同的位数下表现不错，但依然没有繁华能力

```shell
_____________321____#
_____________654_*__#
_____________837____#
______________516___#
_______________294__#
_____________837____#
_____________8886___#
_____________88065__#
____________________#
____________________#
____________________#
```

### SINUSODIAL

为每个位置生成一个固定的正弦余弦交替出现的固定位置编码，加到词向量上。

```python
pe = torch.zeros(config.block_size, config.n_embd)
position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * (-math.log(10000.0) / config.n_embd))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pos_emb = self.sinusoidal_pe[:t, :].unsqueeze(0)
in_emb = pos_emb + tok_emb
```

效果不佳，甚至不如ABS_LEARNED

### CUSTOM_PID

参考[HanseulJo/position-coupling: Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure (NeurIPS 2024) + Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count (ICLR 2025)](https://github.com/HanseulJo/position-coupling)做的二维position_id，每个画布上的每个点有一个2维的位置id，分别代表其行数和列数，两个pid分别做abs_learned的position embedding。

```python
pos_emb_h = self.transformer.wpe_h(pos_h)
pos_emb_w = self.transformer.wpe_w(pos_w)
pos_emb = pos_emb_h + pos_emb_w
in_emb = pos_emb + tok_emb
```

效果最好，在不需要额外添加加法过程即可收敛到基本完全正确，但是泛化能力弱，仅在第一个数字位数增长时体现出一点泛化能力

### 2D-RoPE

在Attention的计算中，对每个head的Q，K矩阵做一个旋转操作，2D的方法是将q，k举证在head_dim上分成两份，分别做RoPE

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,T,D)
k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

# --- Apply 2D RoPE ---
D = self.head_dim
D2 = D // 2
qx, qy = q[..., :D2], q[..., D2:]
kx, ky = k[..., :D2], k[..., D2:]

cos_x = self.cos_x[..., x_pos, :]  # (1,1,T,D2/2)
sin_x = self.sin_x[..., x_pos, :]
cos_y = self.cos_y[..., y_pos, :]
sin_y = self.sin_y[..., y_pos, :]

qx = apply_rope(qx, cos_x, sin_x)
kx = apply_rope(kx, cos_x, sin_x)
qy = apply_rope(qy, cos_y, sin_y)
ky = apply_rope(ky, cos_y, sin_y)

q = torch.cat([qx, qy], dim=-1)
k = torch.cat([kx, ky], dim=-1)
```

效果还行，和custom_pid类似，但泛化能力更弱。

## TinyRecursiveModel

TRM的项目链接位https://github.com/alphaXiv/TinyRecursiveModels
实验的代码位于https://github.com/birdswimming/trm-multi

不同于nanoGPT其输入输出天然就是固定大小的，其在数独，迷宫，ARC-AGI上表现出色。

使用和nanoGPT相同的数据组织形式，同样在一张画布中的任意位置放置一个固定大小的竖式计算的过程，但是每次仅预测竖式计算中的一行而非完整的计算过程。

进行了数据增强，对于任意一对输入输出，其可以进行旋转对称平移等操作变为多组数据。

模型的规模如下

| hidden_size | num_heads | L_layers | H_cycles | L_cycles | 参数总量 |
| ----------- | --------- | -------- | -------- | -------- | -------- |
| 512         | 8         | 2        | 3        | 6        | 80+M     |

在7位数乘7位数的空间内进行训练，10位数乘10位数的空间内进行测试。模型表现出了一定的泛化能力，但依然仅能在第一个数增长时表现出泛化能力。

由于模型还会输出其对答案的置信度，尝试multi-view方法，对于每个输出先进行16倍的数据增强，在选取置信度最高的结果作为最终输出，效果不佳。