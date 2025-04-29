#MoE
#rope
#RMSNorm pre-normalization
#SwiGLU
#kvcache
#Grouped Query Attention (GQA)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
"""
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
ModelArgs = {
    "dim": 512,  # 模型维度
    "inter_dim": 1024,  # 前馈网络的隐藏维度
    "moe_inter_dim": 1024,  # MoE层的隐藏维度
    "n_heads": 8,  # 注意力头数
    "n_layers": 12,  # Transformer层数
    "n_dense_layers": 1, 
    "n_routed_experts": 8,  # 路由专家数量，也就是会被topk的专家
    "n_shared_experts" : 1,  # 共享专家数量
    "n_avtivate_experts": 2,  # 激活的专家数量
    "vocab_size": 50257,  # 词汇表大小
    "max_seq_len": 512,  # 最大序列长度
    "rope_theta": 10000.0,  # 旋转位置编码的参数
    "norm_eps": 1e-6,  # RMSNorm的epsilon值
    "droprate": 0.1,  # Dropout率

}

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 初始化为1的可学习参数

    def _norm(self, x):
        # torch.rsqrt: 平方根的倒数，这里用于计算标准差的倒数
        # x.pow(2).mean(-1, keepdim=True): 沿着倒数第一维计算平方并求平均
        #    a_i * 元素平方的均值取平方根后再取倒数 + 无穷小量
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 计算词向量元素两两分组以后，每组元素对应的旋转角度 
    # torch.arange(0, dim, 2): 生成 [0,2,4...126]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)  # t = [0,....end]
    # torch.outer: torch.outer(a, b) = a^T * b
    freqs = torch.outer(t, freqs)  # freqs.shape = (t.len(),freqs.len()) #shape (end,dim//2)

    # 根据角坐标生成复数向量
    # torch.polar(abs,angle): abs*cos(angle) + abs*sin(angle)*j
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # freqs_cis.shape  = (end,dim//2)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # ndim为x的维度数, 此时应该为4
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) 
    
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # (1, x.shape[1], 1, x.shape[-1])
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将xq和xk的最后一个维度进行复数运算，得到新的xq和xk"""
    # xq.shape = [bsz, seqlen, self.n_local_heads, self.head_dim]
    # xq_.shape = [bsz, seqlen, self.n_local_heads, self.head_dim//2 , 2]
    # torch.view_as_complex用于将二维向量转换为复数域 torch.view_as_complex即([x,y]) -> (x+yj)
    # 所以经过view_as_complex变换后xq_.shape = [bsz, seqlen, self.n_local_heads, self.head_dim//2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # freqs_cis.shape = (1,x.shape[1],1,x.shape[-1])
    
    # xq_ 与freqs_cis广播哈达玛积
    # [bsz, seqlen, self.n_local_heads, self.head_dim//2] * [1,seqlen,1,self.head_dim//2]
    # torch.view_as_real用于将复数再转换回实数向量, 再经过flatten展平第4个维度 
    # [bsz, seqlen, self.n_local_heads, self.head_dim//2] ->[bsz, seqlen, self.n_local_heads, self.head_dim//2,2 ] ->[bsz, seqlen, self.n_local_heads, self.head_dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        assert args["dim"] % args["n_heads"] == 0, "模型维度必须能被头数整除"
        self.args = args
        self.n_heads = args["n_heads"]
        self.d_model = args["dim"]
        self.d_k = self.d_model // self.n_heads  # 每个头的维度
        # self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads else cfg.n_heads
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)  # 查询线性变换
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)  # 键线性变换
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)  # 值线性变换
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)  # 输出线性变换
        self.dropout = nn.Dropout(args["droprate"])  # Dropout层

    def forward(self, x: torch.Tensor, freqs_cis, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_embd = x.shape  # d_model和_相等
        # 线性投影并分割多头
        q = self.w_q(x).view(batch_size,seq_len, self.n_heads, self.d_k)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # 计算旋转位置编码
        q, k = apply_rotary_emb(q, k, freqs_cis)  # q和k的最后一个维度进行复数运算，得到新的q和k
        # 计算注意力分数
        q = q.transpose(1, 2)# (B, H, L, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.d_k)#(B,H,L,K) * (B,H,K,L) -> (B,H,L,L)

        # 应用因果掩码（防止关注未来位置）
        # mask_bool = self.mask.bool()[:seq_len, :seq_len]
        # scores = scores.masked_fill(mask_bool, -torch.inf)

        if mask is not None:
            scores = scores + mask  # mask的形状为(B, L, L)，scores的形状为(B, H, L, L)

        attn_weights = F.softmax(scores/(k.shape[-1]**0.5), dim=-1)  # (B, H, L, L)
        attn_weights = self.dropout(attn_weights)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, v)
        
        #合并多头并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (B, L, d_model)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super(FeedForward, self).__init__()
        # 门控投影层（通常为模型维度的4倍）
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        # 上投影层（通常为模型维度的4倍）
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        # 下投影层（返回原始维度）
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        # 激活函数使用Swish（SiLU）
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播过程：
        1. 门控投影 + 激活函数
        2. 与上投影结果进行逐元素相乘
        3. 下投影回原始维度
        """
        gate = self.act(self.gate_proj(x))  # [batch_size, seq_len, hidden_dim]
        up = self.up_proj(x)                # [batch_size, seq_len, hidden_dim]
        fused = gate * up                   # 逐元素相乘
        return self.down_proj(fused)       # [batch_size, seq_len, dim]


class Expert(nn.Module):
    """单个专家网络（可替换为任意子网络）"""
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.act = nn.SiLU()
        self.w3 = nn.Linear(dim, hidden_dim)
    
    def forward(self, x):
        return self.w2(self.act(self.w1(x)) * self.w3(x)) 

class MoELayer(nn.Module):
    """MoE核心层：动态路由 + 专家并行计算"""
    def __init__(self, args):
        super().__init__()
        self.top_k = args["n_avtivate_experts"]
        self.dim = args["dim"]
        self.n_routed_experts = args["n_routed_experts"]
        # 专家池
        self.experts = nn.ModuleList(
            [Expert(args["dim"], args["moe_inter_dim"]) for _ in range(args["n_routed_experts"])]
        )
        self.shared_experts = FeedForward(args["dim"], args["moe_inter_dim"]* args["n_shared_experts"])
        # 门控网络
        self.gate = nn.Linear(args["dim"], args["n_routed_experts"])
        
        # 负载平衡统计量
        self.register_buffer('avg_probs', torch.zeros(args["n_routed_experts"]))

        self.bias = None
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        shape = x.size()
        x = x.view(-1, self.dim)
        
        # Step 1: 计算门控权重
        gate_logits = self.gate(x)
        scores = F.softmax(gate_logits, dim=-1)

        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias

        # Step 2: 选择Top-K专家
        topk_probs, indices = torch.topk(
            scores, k=self.top_k, dim=-1
        )
        
        weights = original_scores.gather(1, indices) 
        weights.type_as(x)

        shared_expert_output = self.shared_experts(x)  # [batch * seq_len, dim]
        router_expert_output= torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(len(self.experts)):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            router_expert_output[idx] += expert(x[idx]) * weights[idx, top, None]
        
        return (shared_expert_output + router_expert_output).view(shape)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args):
        super(TransformerBlock, self).__init__()
        self.attn = Attention(args)
        self.ffn = FeedForward(args["dim"], args["inter_dim"]) if layer_id < args["n_dense_layers"] else MoELayer(args)
        self.attn_norm = RMSNorm(args["dim"])
        self.ffn_norm = RMSNorm(args["dim"])

    def forward(self, x: torch.Tensor, freqs_cis, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Tunx(nn.Module):
    def __init__(self, args = ModelArgs):
        super(Tunx, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args["vocab_size"], args["dim"])
        self.layers = nn.ModuleList([TransformerBlock(idx, args) for idx in range(args["n_layers"])])
        self.norm = RMSNorm(args["dim"])
        self.head = nn.Linear(args["dim"], args["vocab_size"], bias=False)  # 输出层
        self.freqs_cis = precompute_freqs_cis(args["dim"]//args["n_heads"], args["max_seq_len"] * 2, args["rope_theta"])
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.shape
        # 词嵌入
        x = self.embedding(x)

        freqs_cis = self.freqs_cis[:seq_len].to(x.device)  # 计算旋转位置编码

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device).triu_(1)
        
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        x = self.norm(x)
        x = self.head(x)
        return x