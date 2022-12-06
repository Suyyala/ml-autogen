class TransformerEncoderLayer(torch.nn.Module):
  def __init__(self, emb_dim, num_heads, ff_dim, dropout):
    # Initialize the multi-head attention layer
    self.attention = MultiHeadAttention(emb_dim, num_heads, dropout)
    # Initialize the feed-forward layer
    self.feed_forward = FeedForward(emb_dim, ff_dim, dropout)
    # Initialize the layer normalization layer
    self.layer_norm = torch.nn.LayerNorm(emb_dim)
    # Initialize the dropout layer
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, inputs, mask):
    # Pass the input sequence through the multi-head attention layer
    attention_output = self.attention(inputs, inputs, inputs, mask)
    # Add the attention output to the input sequence
    attention_output = inputs + self.dropout(attention_output)
    # Pass the combined sequence through the feed-forward layer
    feed_forward_output = self.feed_forward(attention_output)
    # Add the feed-forward output to the combined sequence
    feed_forward_output = attention_output + self.dropout(feed_forward_output)
    # Pass the output sequence through the layer normalization layer
    output = self.layer_norm(feed_forward_output)
    return output
