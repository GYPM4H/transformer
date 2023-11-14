import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a long enough 'position' tensor
        position = torch.arange(max_len).unsqueeze(1)
        # Use the div term to create a tensor of shape (max_len, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        # Register as a buffer that is not a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input embedding tensor
        x = x + self.pe[:,:x.size(0), :]
        return x
    

# d_model = 16
# max_len = 50

# pos_enc = PositionalEncoding(d_model, max_len)

# input_tensor = torch.rand(1, max_len, d_model)
# output = pos_enc(input_tensor)

# # print(output.shape)