import torch
import torch.nn as nn
from MaskedMultiHeadAttention import MaskedMultiHeadAttention
from PositionalEncoder import PositionalEncoding

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.multi_head_attention = MaskedMultiHeadAttention(d_model, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Position-wise Feedforward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, src):
        attn_output, attention_weights = self.multi_head_attention(src, src, src)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        
        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        
        return src, attention_weights

# d_model = 16
# num_heads = 2
# input_vector = torch.rand(1, 200, d_model)

# encoder_layer = TransformerEncoderLayer(d_model, num_heads)
# encoder_output = encoder_layer(input_vector)

# print(encoder_output.shape)