import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(max_seq_length, model_size):
    """Function to create positional encodings for transformers.
    
    Args:
    max_seq_length: The maximum sequence length.
    model_size: The size of the model (i.e., the size of the embeddings).
    
    Returns:
    A numpy array of shape (max_seq_length, model_size) containing the positional encodings.
    """
    # Initialize a matrix of shape (max_seq_length, model_size) with positional encodings
    pos_enc = np.zeros((max_seq_length, model_size))
    
    # Compute the positional encodings using sine and cosine functions
    for pos in range(max_seq_length):
        for i in range(0, model_size, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / model_size)))
            if i + 1 < model_size:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / model_size)))
    
    return pos_enc

# Let's create positional encodings for a model with embedding size of 512 and max sequence length of 50.
max_seq_length = 50
model_size = 512

pos_enc = positional_encoding(max_seq_length, model_size)

# Now let's plot the positional encodings to see how they vary for different dimensions
plt.figure(figsize=(15,10))
plt.plot(pos_enc[:, :64])  # Let's plot the first 64 dimensions
plt.xlabel('Position')
plt.ylabel('Encoding value')
plt.title('Positional Encoding for Various Dimensions')
plt.grid(True)
plt.show()
