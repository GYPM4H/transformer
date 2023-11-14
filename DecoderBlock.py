import torch
from torch import nn

from PositionalEncoder import PositionalEncoding
from MaskedMultiHeadAttention import MaskedMultiHeadAttention

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, encoder_out, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.encoder_out = encoder_out

        self.masked_multi_head_attention = MaskedMultiHeadAttention(d_model, num_heads, mask=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.multi_head_attention = MaskedMultiHeadAttention(d_model, num_heads, mask=False)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
        nn.Linear(d_model, dim_feedforward),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        attn_out_1, attn_weights_1 = self.masked_multi_head_attention(src, src, src)
        src = src + self.dropout(attn_out_1)
        src = self.norm1(src)

        attn_out_2, attn_weights_2 = self.multi_head_attention(src, self.encoder_out, self.encoder_out)
        src = src + self.dropout(attn_out_2)
        src = self.norm2(src)

        ff_out = self.feed_forward(src)
        src = src + self.dropout(ff_out)
        src = self.norm3(src)

        return src, attn_weights_1, attn_weights_2
