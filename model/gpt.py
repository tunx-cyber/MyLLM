import torch
import torch.nn as nn
import torch.nn.functional as F
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

'''
Q1:既然滑动窗口已经控制了下标，为什么还需要 attention_mask？
答案：滑动窗口解决了数据切割问题，但无法解决以下问题：

尾部填充需求：当剩余文本不足窗口大小时必须填充。

批处理对齐：同一批次内样本需统一长度。

模型计算需求：自注意力机制需要明确无效位置。

attention_mask 是填充掩码，用于标识有效数据区域。必须手动生成，因为：

每个样本的填充长度不同

滑动窗口切割可能导致新的填充需求
Q2：能否完全避免填充？
理论可能：若所有文本长度恰好是窗口大小的整数倍，且使用动态batch策略。

现实情况：实际数据长度分布复杂，动态batch会显著降低训练效率（显存碎片化、无法使用Tensor Core加速）。
'''

class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512,drop_rate = 0.1,qkv_bias=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model  # 模型维度
        self.n_heads = n_heads  # 注意力头数
        self.d_k = d_model // n_heads  # 每个头的维度

        # 线性投影层（Q/K/V/O）
        self.w_q = nn.Linear(d_model, d_model,bias=qkv_bias)  # 查询投影
        self.w_k = nn.Linear(d_model, d_model,bias=qkv_bias)  # 键投影
        self.w_v = nn.Linear(d_model, d_model,bias=qkv_bias)  # 值投影
        self.w_o = nn.Linear(d_model, d_model)  # 输出投影
        self.register_buffer("mask", torch.triu(
            torch.ones(max_len, max_len), diagonal=1)
        )  # 注册掩码缓冲区
        self.dropout = nn.Dropout(drop_rate)  # Dropout层

    def forward(self, x):
        """
        输入:
            x: 输入张量 (batch_size, seq_len, d_model)
            mask: 因果掩码 (batch_size, 1, seq_len, seq_len)
        输出:
            注意力后的张量 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape #d_model和_相等

        # 1. 线性投影并分割多头 把_也就是d_model分为head*d_k
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L, d_k)

        # causal_mask = self.mask.bool()[:seq_len, :seq_len]  # (L, L)
        # causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        # causal_mask = causal_mask.expand(batch_size, self.n_heads, -1, -1)  # (B, H, L, L)

        # padding_mask = attention_mask
        # if padding_mask is not None:
        #     # 转换填充掩码为注意力掩码格式
        #     padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
        #     padding_mask = padding_mask.expand(-1, self.n_heads, seq_len, -1)  # [B, H, L, L]
            
        #     # 合并逻辑：如果任一掩码为True（需要屏蔽），则最终掩码为True
        #     combined_mask = causal_mask | (~padding_mask)  # [B, H, L, L]
        # else:
        #     combined_mask = causal_mask

        # 2. 计算缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1))# (B, H, L, d_k) * (B, H, d_k, L) = (B, H, L, L)
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        # 3. 应用因果掩码（防止关注未来位置）
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)  # (B, H, L, L)

        attn_weights = F.softmax(attn_scores/(k.shape[-1]**0.5), dim=-1)  # (B, H, L, L)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)    #(B, H, L, L) * (B, H, L, d_k) = (B, H, L, d_k)

        # 4. 合并多头并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (B, L, d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super(FeedForward, self).__init__()  
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self,x):
        x = self.layers(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadCausalAttention(
            cfg["emb_dim"], 
            n_heads=cfg["n_heads"], 
            max_len=cfg["max_length"],
            drop_rate=cfg["drop_rate"], 
            qkv_bias=cfg["qkv_bias"])
        self.feed_forward = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        #prelayer norm
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut
        return x

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
        for trf_block in self.trf_blocks:
            x = trf_block(x)
        # x = self.trf_blocks(x,attn_mask)
        x = self.final_norm(x) # (B, L, d_model)
        logits = self.out_head(x) # (B, L, vocab_size)
        return logits