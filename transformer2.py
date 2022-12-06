import torch
from torch import nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dropout):
        super().__init__()
        
        # initialize the encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # final linear layer to produce the output
        self.linear = nn.Linear(d_model, vocab_size)
        
        # attention layer
        self.attn = MultiHeadAttention(n_heads, d_model)
        
    def forward(self, src, src_mask):
        # pass the input through the encoder layers
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        # apply attention to generate a continuous representation of the input
        cont_rep, attn_weights = self.attn(src, src, src, src_mask)
        
        # apply the final linear layer and return the output
        return self.linear(cont_rep), attn_weights

      
      import torch


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, dropout):
        super().__init__()
        
        # initialize the decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # final linear layer to produce the output
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, trg, enc_out, src_mask, trg_mask, cont_rep):
        # pass the input through the decoder layers
        for layer in self.decoder_layers:
            trg = layer(trg, enc_out, src_mask, trg_mask)
        
        # concatenate the continuous representation of the input with the output
        # of the decoder layers
        trg = torch.cat((cont_rep, trg), dim=1)
        
        # apply the final linear layer and return the output
        return self.linear(trg)
