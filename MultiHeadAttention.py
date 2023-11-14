import torch
from torch import nn
from MaskedMultiHeadAttention import MaskedMultiHeadAttention
# Let's implement a simple version of the splitting process for multi-head attention.
torch.manual_seed(0)

class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SimpleMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Ensure the model dimension is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // num_heads
        
        # These are the weight matrices for queries, keys, and values
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        # This final linear layer will combine the outputs from all heads
        self.final_linear = nn.Linear(d_model, d_model)
    
    def split_into_heads(self, x):
        # Split the last dimension into (num_heads, depth)
        new_shape = x.size()[:-1] + (self.num_heads, self.depth)
        x = x.view(new_shape)
        # Permute to get dimensions (batch_size, num_heads, seq_len, depth)
        return x.permute(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, q, k, v):
        k = k.permute(0, 1, 3, 2) # Transpose
        dot_product = torch.matmul(q, k)
        scale = torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        attention_weights = torch.softmax(dot_product / scale, dim=-1)

        out = torch.matmul(attention_weights, v)
        return out, attention_weights
    
    def forward(self, q, k, v):
        # Apply the linear transformations to get Q, K, V
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # Split Q, K, V into multiple heads
        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        attention_output, attention_weights = self.scaled_dot_product_attention(q, k, v)

        # We need to transpose back to (batch_size, seq_len, num_heads, depth)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()

        # Now we can flatten the last two dimensions
        batch_size, seq_len, _, _ = attention_output.size()
        attention_output = attention_output.view(batch_size, seq_len, -1)

        # Apply the final linear layer
        out = self.final_linear(attention_output)

        return out, attention_weights

# # Create an instance of our simple multi-head attention module
d_model = 16
num_heads = 2

input = torch.rand(1, 5, d_model)
smha = SimpleMultiHeadAttention(d_model, num_heads)
mmha = MaskedMultiHeadAttention(d_model, num_heads)

out, weights = smha(input, input, input)
out1, weights1 = mmha(input, input, input)

print(out1.shape), print(weights1.shape)
print(out.shape), print(weights.shape)
sim = torch.norm(out1 - out).item()
print(sim)


