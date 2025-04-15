import torch
import torch.nn as nn
from .transformer import TransformerBlock
GPT_CONFIG_124M = {
    "vocab_size": 50257, # 词汇表大小
    "max_length": 1024, # 上下文长度
    "emb_dim": 768, # 嵌入维度
    "n_heads": 12,# 注意力头数
    "n_layers": 12,# 层数 transformer
    "drop_rate": 0.1,# Dropout率
    "qkv_bias": False,
    "layer_sizes": [768, 768*4, 768], # 前馈网络的层大小
    "use_res": True,
}
'''
准备输入文本以进行LLM训练的最后一步是将词ID转换为嵌入向量
嵌入层的权重矩阵包含小的随机值。这些值会在LLM训练过程中进行优化。
此外，我们可以看到权重矩阵有六行和三列。
每一行对应词汇表中的六个可能词之一，每一列对应三个嵌入维度之一。
这里描述的嵌入层方法本质上只是实现独热编码后进行矩阵乘法的一种更高效的方式

每个输出矩阵中的每一行都是通过从嵌入权重矩阵中查找得到的
'''

'''
相对位置嵌入和绝对位置嵌入
OpenAI的GPT模型使用的是在训练过程中优化的绝对位置嵌入
'''

'''
注意力机制
注意力机制的核心思想是通过计算查询（query）和键（key）之间的相似度来决定
模型在处理输入时应该关注哪些部分。具体来说，注意力机制通过计算查询和键之间的点积来衡量它们之间的相似度。
点积是一个数学操作，它将两个向量组合成一个标量值。它的计算方法是将两个向量
对应位置的元素相乘，然后将这些乘积相加。点积的结果可以是正数、负数或零，具体取决于两个向量之间的关系。
在深度学习中，点积通常用于计算两个向量之间的相似度或相关性。它可以用
于比较两个向量的方向和大小。例如，在自然语言处理任务中，点积可以用于计算
单词嵌入之间的相似度，从而帮助模型理解单词之间的关系。
在自注意力机制中，点积用于计算查询和键之间的相似度，从而决定模型在处理
输入时应该关注哪些部分。具体来说，模型会计算每个查询与所有键之间的点积，然后
将这些点积值进行归一化处理，得到注意力权重。最后，模型会根据这些注意力权重
对值进行加权求和，从而得到最终的输出。

1. compute attention scores
2. compute attention weights
3. compute context vectors

查询类似于数据库中的搜索查询。它代表了模型当前关注或试图理解的项目（例
如句子中的一个词或标记）。查询用于探测输入序列的其他部分，以确定应给予
它们多少注意力。
键类似于用于索引和搜索的数据库键。在注意力机制中，输入序列中的每个项目
（例如句子中的每个词）都有一个关联的键。这些键用于匹配查询。
在这个上下文中，值类似于数据库中的键值对中的值。它表示输入项目的实际内
容或表示。一旦模型确定哪些键（因此是输入的哪些部分）与查询（当前关注
项）最相关，它就会检索相应的值。
'''
'''
应用因果注意力掩码
在自回归模型中，我们需要确保每个位置只能看到它之前的位置，以保持因果关系。
在实现中，我们可以使用一个上三角矩阵来实现这个掩码。具体来说，我们可以创建一个大小为
(seq_len, seq_len)的上三角矩阵，并将其与注意力权重矩阵相乘。
这样，只有上三角矩阵中对应位置为1的元素会被保留，而其他位置会被置为0。
在实现中，我们可以使用torch.triu函数来创建上三角矩阵，并将其与注意力权重矩阵相乘。
'''

class MyGPT(nn.Module):
    def __init__(self,cfg):
        super(MyGPT, self).__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["max_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"],bias=False)

    def forward(self,in_idx):
        device = in_idx.device
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.token_embedding(in_idx)
        pos_embeds = self.position_embedding(torch.arange(seq_len, device=device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x) # (B, L, d_model)
        logits = self.out_head(x) # (B, L, vocab_size)
        return logits