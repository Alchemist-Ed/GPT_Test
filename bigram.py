
import tiktoken
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F



                                                            ####################################
                                                            ####### 参数声明 ####################
                                                            ####################################


batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#################


torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#print('Length of dataset in characters is', len(text))
#print(text[:50])

                                                                    ####################
                                                                    ### tokenization ###
                                                                    ####################

#### check unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

# 第一步，enumerate 文本所有字符的集合, chars, 找到每个字母和其对应的index
stoi = {ch:i for i, ch in enumerate(chars)}
# 同时准备decode的字典，即所有index和其对应的文本
itos = {i:ch for i, ch in enumerate(chars)}

## encoder将字符转换成索引(int), 而decoder将索引转换回字符，所以需要两个key,value相反的字典

## encoder函数，输出输入文本中，每个字符对应的索引
encode = lambda s: [stoi[c] for c in s]
## decoder函数，输出输入字符中，每个整数对应的字符
decode = lambda l: ''.join([itos[i] for i in l])


### 直接调用tiktoken lib里的encoder/decoder函数也可
# enc = tiktoken.get_encoding('gpt2')
# print(enc.n_vocab)
# print(enc.encode('Hello World'))
# print(enc.decode([15496, 2155]))


###################
### 向量化token ####
###################

data = torch.tensor(encode(text), dtype=torch.long)
#print(data.shape, data.dtype)
#print(data[:10])

#### 构建训练和验证数据集
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


#### Block size/context size， 一条输入文本中最多包含多少token
#### Batch size，有多少条独立的文本同时被并行训练

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    ### 使用cuda时，确保数据被转移到cuda的设备上
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')


##############################
######## 特制损失函数 #########
##############################

### 使用no_grad()函数，当目标函数不会进行反向传播运算时，这样目标函数将不会自动储存运算结果
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


##########################
#### 生成模型 #############
##########################

### 这里客制化一个最基础的语言模型，继承于nn.Module类
class BigramLanguageModel(nn.Module):
    ## 改写init函数，初始化一个嵌入层
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    ## 改写forward函数，向前传播方法改为查询输入值在嵌入层中的对应向量
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)
## 创建模型时，模型也需要转移到device上

## 这里存在nn.Module类的隐式调用，调用的是forward函数
logits, loss = m(xb, yb)
## 手动设置第一个输入的向量，设置为([0,0])的张量，代表unicode中的start符号

## 现阶段的模型并没有被训练，生成结果全部是随机数
#print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


##########################
#### 训练模型 #############
##########################


############ 设置优化器 #############
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

#### 优化损失

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step{iter}: train loss{losses['train']:.4f}, val loss {losses['val']:.4f}')

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    ## 每个新批次开始前，重置梯度为0，避免梯度累积
    optimizer.zero_grad(set_to_none=True)
    ## 对损失使用反向传播计算
    loss.backward()
    ## 根据优化器所使用的算法来更新模型参数
    optimizer.step()
    print(loss.item())
    
### 生成预测文本时，也需要再device上生成
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
