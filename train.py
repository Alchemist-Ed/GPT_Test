
import tiktoken
import torch
import numpy as np

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
print(data.shape, data.dtype)
print(data[:10])

#### 构建训练和验证数据集

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
