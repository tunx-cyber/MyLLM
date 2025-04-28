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

class ModelArgs:
    dim: int = 4096  # 模型维度
    n_layers: int = 32  # 层数
    n_heads: int = 32  # 头数
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # 词汇表大小
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048  # 序列长度

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
    def __init__(self, cfg: ModelArgs):
        super(Attention, self).__init__()
        assert cfg.dim % cfg.n_heads == 0, "模型维度必须能被头数整除"
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_model = cfg.dim
        self.d_k = self.d_model // self.n_heads  # 每个头的维度
        # self.n_kv_heads = cfg.n_kv_heads if cfg.n_kv_heads else cfg.n_heads
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)  # 查询线性变换
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)  # 键线性变换
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)  # 值线性变换
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)  # 输出线性变换
        # self.mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1)
        self.dropout = nn.Dropout(cfg.droprate)  # Dropout层

        # self.freqs_cis = precompute_freqs_cis(self.d_model, cfg.max_seq_len * 2)
    
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
            scores = scores + mask.unsqueeze(1)  # mask的形状为(B, 1, L, L)，scores的形状为(B, H, L, L)

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
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    """MoE核心层：动态路由 + 专家并行计算"""
    def __init__(self, num_experts, input_dim, output_dim, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 专家池
        self.experts = nn.ModuleList(
            [Expert(input_dim, output_dim) for _ in range(num_experts)]
        )
        # 门控网络
        self.gate = nn.Linear(input_dim, num_experts)
        
        # 负载平衡统计量
        self.register_buffer('avg_probs', torch.zeros(num_experts))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Step 1: 计算门控权重
        gate_logits = self.gate(x)  # [batch, seq_len, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Step 2: 选择Top-K专家
        topk_probs, topk_indices = torch.topk(
            gate_probs, k=self.top_k, dim=-1)  # [batch, seq_len, top_k]
        
        # Step 3: 生成路由掩码（稀疏计算）
        mask = F.one_hot(topk_indices, self.num_experts)  # [batch, seq_len, top_k, num_experts]
        mask = mask.sum(dim=2)  # [batch, seq_len, num_experts]
        
        # Step 4: 并行计算所有专家输出
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.num_experts, -1)  # [batch, seq_len, num_experts, input_dim]
        expert_outputs = torch.stack([expert(x_expanded[..., i, :]) 
                                     for i, expert in enumerate(self.experts)], dim=-2)
        # expert_outputs shape: [batch, seq_len, num_experts, output_dim]
        
        # Step 5: 加权聚合结果
        weighted_output = (expert_outputs * topk_probs.unsqueeze(-1)).sum(dim=2)
        
        # 更新负载平衡统计量（用于后续损失计算）
        self.avg_probs = 0.9 * self.avg_probs + 0.1 * gate_probs.mean(dim=(0,1)).detach()
        
        return weighted_output
    
    def load_balancing_loss(self):
        """负载平衡损失（防止专家被忽略）"""
        prob_mean = self.avg_probs.mean()
        return self.num_experts * (self.avg_probs * prob_mean.log()).sum()

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelArgs):
        super(TransformerBlock, self).__init__()
        self.attention = Attention(cfg)
        self.ffn = FeedForward(cfg.dim, cfg.inter_dim) if layer_id < cfg.n_dense_layers else MoELayer(cfg)
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)

    def forward(self, x: torch.Tensor, freqs_cis, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Tunx(nn.Module):
    def __init__(self, cfg: ModelArgs):
        super(Tunx, self).__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([TransformerBlock(_, cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)  # 输出层
        self.freqs_cis = precompute_freqs_cis(cfg.dim, cfg.max_seq_len * 2, cfg.rope_theta)
    
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