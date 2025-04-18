import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, x, attention_mask = None):
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

        causal_mask = self.mask.bool()[:seq_len, :seq_len]  # (L, L)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.expand(batch_size, self.n_heads, -1, -1)  # (B, H, L, L)

        padding_mask = attention_mask
        if padding_mask is not None:
            # 转换填充掩码为注意力掩码格式
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, L]
            padding_mask = padding_mask.expand(-1, self.n_heads, seq_len, -1)  # [B, H, L, L]
            
            # 合并逻辑：如果任一掩码为True（需要屏蔽），则最终掩码为True
            combined_mask = causal_mask | (~padding_mask)  # [B, H, L, L]
        else:
            combined_mask = causal_mask

        # 2. 计算缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1))# (B, H, L, d_k) * (B, H, d_k, L) = (B, H, L, L)

        # 3. 应用因果掩码（防止关注未来位置）
        attn_scores = attn_scores.masked_fill(combined_mask, -1e9)

        attn_weights = F.softmax(attn_scores/(k.shape[-1]**0.5), dim=-1)  # (B, H, L, L)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)    #(B, H, L, L) * (B, H, L, d_k) = (B, H, L, d_k)

        # 4. 合并多头并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (B, L, d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, layer_sizes, use_res):
        super(FeedForward, self).__init__()
        self.use_res = use_res
        self.layers =  nn.ModuleList(
            [nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i+1]),nn.GELU())
            for i in range(len(layer_sizes)-1)]
        )
    
    def forward(self,x):
        for layer in self.layers:
            output = layer(x)
            if self.use_res and x.shape == output.shape:
                x = output + x
            else:
                x = output
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
        self.feed_forward = FeedForward(cfg["layer_sizes"], False)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, attention_mask=None):
        attention = self.attention(x,attention_mask)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out#(B, L, d_model)
    
