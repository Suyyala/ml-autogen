import torch
from torch import nn

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        
        # create a matrix of sinusoids with shape (max_seq_len, d_model)
        position_encoding = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                position_encoding[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                position_encoding[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        position_encoding = position_encoding.unsqueeze(0)
        
        # register the matrix of sinusoids as a buffer
        self.register_buffer('position_encoding', position_encoding)
        
    def forward(self, seq):
        # add the sinusoids to the input tensor
        seq = seq + self.position_encoding[:, :seq.size(1)]
        return seq
