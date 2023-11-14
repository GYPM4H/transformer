import numpy
import torch
import torch.nn as nn

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, mask = False):
        super(MaskedMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.mask = mask 
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.depth = d_model // num_heads
        
        self.wqs = nn.ModuleList(nn.Linear(self.d_model, self.depth) for _ in range(self.num_heads))
        self.wks = nn.ModuleList(nn.Linear(self.d_model, self.depth) for _ in range(self.num_heads))
        self.wvs = nn.ModuleList(nn.Linear(self.d_model, self.depth) for _ in range(self.num_heads))

        self.final_linear = nn.Linear(self.d_model, self.d_model)

    def scaled_dot_product_attention(self, q, k, v):
        attention_weights = torch.matmul(q, k.transpose(-2, -1))
        attention_weights = attention_weights / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        
        if self.mask != False:
            mask = torch.ones(attention_weights.size()[-2], attention_weights.size()[-1]).triu(diagonal=1)
            attention_weights = attention_weights.masked_fill(mask == 1, float('-inf'))

        attention_weights = torch.softmax(attention_weights, dim=-1)
        out = torch.matmul(attention_weights, v)
        return out, attention_weights

    def forward(self, q, k, v):
        attention_out = []
        attention_weights = []
        for i in range(self.num_heads):
            q_head = self.wqs[i](q)
            k_head = self.wks[i](k)
            v_head = self.wvs[i](v)

            out, weights = self.scaled_dot_product_attention(q_head, k_head, v_head)
            attention_out.append(out)
            attention_weights.append(weights)

        out = torch.cat(attention_out, dim=-1)
        attention_weights = torch.stack(attention_weights, dim=0).permute(1, 0, 2, 3).contiguous()

        out = self.final_linear(out)

        return out, attention_weights
        


import matplotlib.pyplot as plt
import seaborn as sns

d_model = 16
num_heads = 2


input = torch.rand(1, 5, d_model)
mmha = MaskedMultiHeadAttention(d_model, num_heads, mask=True)
out, weights = mmha(input, input, input)

print(out.shape), print(weights.shape)

head = 0 
attention_matrix = weights[0, head].detach().numpy()

# Use seaborn to create a heatmap
sns.heatmap(attention_matrix, cmap='Reds')

plt.title('Attention Heatmap for Head {}'.format(head))
plt.xlabel('Key')
plt.ylabel('Query')
plt.show()
