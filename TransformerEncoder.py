import torch
import torch.nn as nn

from EncoderBlock import TransformerEncoderLayer
from PositionalEncoder import PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dim_feedforward=2048, dropout=0.1, max_len=5000):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=dropout)

    def __repr__(self):
        return "TransformerEncoder(vocab_size={}, d_model={}, num_heads={}, num_layers={}, dim_feedforward={}, dropout={}, max_len={})".format(
            self.vocab_size, self.d_model, self.num_heads, self.num_layers, self.dim_feedforward, self.dropout, self.max_len
        )

    def forward(self, src):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model))
        src = self.pos_encoder(src)

        attention_weights = None
        for layer in self.layers:
            src, attention_weights = layer(src)

        return src, attention_weights
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    vocab = {
        "Do": 0,
        "I": 1,
        "wanna": 2,
        "know": 3,
    }

    vocab_size = len(vocab)
    d_model = 16
    num_heads = 2
    num_layers = 2

    sentence = "Do I wanna know"
    
    tokens = sentence.split()
    token_indices = torch.tensor([[vocab[token] for token in tokens]], dtype=torch.long)

    encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers)
    print(encoder)

    print(f"Input sentence is: {sentence}")
    print(f"Token indices are: {token_indices}")

    output, attention_weights = encoder(token_indices)

    print(f"Output shape is: {output.shape}")
    print(f"Output is: {output}")
    print(f"Attention weights shape is: {attention_weights.shape}")

    head = 0
    attention_matrix = torch.mean(attention_weights[0], dim= 0).detach().numpy()

    # Use seaborn to create a heatmap
    sns.heatmap(attention_matrix, xticklabels=[token for token in vocab], yticklabels=[token for token in vocab], cmap='Blues')

    plt.title(f'Attention heatmap averaged over {num_heads} heads')
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.show()