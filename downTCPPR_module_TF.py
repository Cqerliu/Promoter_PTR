import torch
import torch.nn.functional as F
from torch import nn, einsum
from entmax import Sparsemax
from einops import rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.selector = Sparsemax(dim=-1)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.selector(sim)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class CrossAttention(nn.Module):

    def __init__(self, dim, heads=5, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)  
        self.to_k = nn.Linear(dim, inner_dim, bias=False)  
        self.to_v = nn.Linear(dim, inner_dim, bias=False)  
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.selector = Sparsemax(dim=-1)
        self.attn_weights = None  

    def forward(self, fc, ft):

        h = self.heads
        q = self.to_q(fc)  # (B, seq_len_fc, inner_dim)
        k = self.to_k(ft)  # (B, seq_len_ft, inner_dim)
        v = self.to_v(ft)  # (B, seq_len_ft, inner_dim)

        # 拆分多头
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)  # (B, heads, seq_len_ft, dim_head)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)  # (B, heads, seq_len_fc, dim_head)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)  # (B, heads, seq_len_fc, dim_head)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.selector(sim)
        self.attn_weights = attn 
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # (B, heads, seq_len_ft, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)  # (B, seq_len_ft, inner_dim)
        return self.to_out(out)  # (B, seq_len_ft, dim)


class TransformerFeature(nn.Module):
    def __init__(self,
                 input_dim,
                 max_seq_len,
                 depth=3,
                 heads=8,
                 dim_head=16,
                 attn_dropout=0.1,
                 ff_dropout=0.1,
                 use_projection=True
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.use_projection = use_projection

   
        if use_projection:
            self.projection = nn.Linear(input_dim, input_dim)


        self.pos_embeds = nn.Embedding(max_seq_len, input_dim)

       
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(input_dim, Attention(
                    input_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout
                ))),
                Residual(PreNorm(input_dim, FeedForward(input_dim, dropout=ff_dropout))),
            ]))

    def forward(self, x):

        B, seq_len, _ = x.shape

 
        if self.use_projection:
            x = self.projection(x)

       
        pos_indices = torch.arange(seq_len, device=x.device)  # (seq_len,)
        pos_emb = self.pos_embeds(pos_indices).unsqueeze(0)  # (1, seq_len, input_dim)
        x = x + pos_emb  # (B, seq_len, input_dim)

     
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

    
        return x  # (B, seq_len_ft, input_dim)


class CNNFeature(nn.Module):
    def __init__(self,
                 input_length,
                 input_channels=1,
                 feature_dim=128,
                 conv_filters=[32, 64],
                 conv_kernels=[3, 3],
                 pool_sizes=[2, 2],
                 dropout_rate=0.3):
        super().__init__()
        self.input_length = input_length
        self.feature_dim = feature_dim

        layers = []
        in_channels = input_channels

        for i in range(len(conv_filters)):
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv_filters[i],
                kernel_size=conv_kernels[i],
                padding='same'  
            ))
            layers.append(nn.BatchNorm1d(conv_filters[i]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=pool_sizes[i]))  
            layers.append(nn.Dropout(dropout_rate))
            in_channels = conv_filters[i]

        self.conv_layers = nn.Sequential(*layers)

        self.channel_proj = nn.Conv1d(
            in_channels=conv_filters[-1],
            out_channels=feature_dim,
            kernel_size=1  
        )

    def forward(self, x):

        conv_out = self.conv_layers(x)  

       
        conv_out = self.channel_proj(conv_out)
        return conv_out.permute(0, 2, 1)


class FeatureFusion(nn.Module):
    def __init__(self, feature_dim, heads=2, dim_head=32, dropout=0.2):
        super().__init__()
        self.feature_dim = feature_dim

        self.cross_attn_ft2fc_1 = PreNorm(feature_dim, CrossAttention(
            dim=feature_dim, heads=heads, dim_head=dim_head, dropout=dropout
        ))
        self.cross_attn_ft2fc_2 = PreNorm(feature_dim, CrossAttention(
            dim=feature_dim, heads=heads, dim_head=dim_head, dropout=dropout
        ))

      
        self.self_attn_rnap = PreNorm(feature_dim, Attention(
            dim=feature_dim, heads=heads, dim_head=dim_head, dropout=dropout
        ))
        self.self_attn_promoter = PreNorm(feature_dim, Attention(
            dim=feature_dim, heads=heads, dim_head=dim_head, dropout=dropout
        ))

   
        self.projection = nn.Sequential(
            nn.Linear(1 * feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, promoter_features, rnap_features, return_3d=True):

       
        rnap_self = self.self_attn_rnap(rnap_features) + rnap_features  # (B, seq_len_rnap, dim)
        promoter_self = self.self_attn_promoter(promoter_features) + promoter_features  # (B, seq_len_prom, dim)

      
        promoter_cross1 = self.cross_attn_ft2fc_1(promoter_self, rnap_self)  + promoter_self
        attn_weights1 = self.cross_attn_ft2fc_1.fn.attn_weights

        fused = promoter_cross1
        return self.projection(fused), attn_weights1


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_dims=[128, 64],
                 activation='relu',
                 dropout=0.2,
                 batch_norm=False
                 ):
        super().__init__()

        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"not support: {activation}")

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)