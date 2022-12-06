class PositionalEncoder(torch.nn.Module):
  def __init__(self, emb_dim):
    # Initialize the position embedding tensor
    self.position_embedding = torch.nn.Embedding(MAX_SEQ_LEN, emb_dim)
    # Initialize the dropout layer
    self.dropout = torch.nn.Dropout(DROPOUT)

  def forward(self, mask):
    # Generate the position indices for each token in the sequence
    position_ids = torch.arange(mask.size(-1), dtype=torch.long, device=mask.device)
    position_ids = position_ids.unsqueeze(0).expand_as(mask)
    # Extract the positions of the non-padding tokens
    positions = position_ids[mask]
    # Look up the position embeddings for the non-padding tokens
    position_embeddings = self.position_embedding(positions)
    # Apply dropout to the position embeddings
    position_embeddings = self.dropout(position_embeddings)
    # Replace the padding tokens in the sequence with zeros
    position_embeddings = position_embeddings * mask.unsqueeze(-1).type_as(position_embeddings)
    return position_embedd
