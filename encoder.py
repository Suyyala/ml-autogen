import torch
from torch import nn
from transformers import TransformerEncoder, TransformerEncoderLayer

class Encoder(nn.Module):
    def __init__(self, num_layers=24, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()

        # Define the encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Define the final linear layer
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, input_ids, attention_mask=None):
        # Pass the input through the encoder layers
        encoder_output = input_ids
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, attention_mask=attention_mask)

        # Apply the final linear layer to generate the context vectors
        context_vectors = self.linear(encoder_output)

        return context_vectors
