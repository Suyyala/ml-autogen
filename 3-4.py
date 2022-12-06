class TransformerDecoderLayer(torch.nn.Module):
  def __init__(self, emb_dim, num_heads, ff_dim, dropout):
    # Initialize the multi-head attention layer that attends to the target sequence
    self.self_attention = MultiHeadAttention(emb_dim, num_heads, dropout)
    # Initialize the multi-head attention layer that attends to the encoder outputs
    self.encoder_attention = MultiHeadAttention(emb_dim, num_heads, dropout)
    # Initialize the feed-forward layer
    self.feed_forward = FeedForward(emb_dim, ff_dim, dropout)
    # Initialize the layer normalization layer
    self.layer_norm = torch.nn.LayerNorm(emb_dim)
    # Initialize the dropout layer
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, inputs, encoded_input, input_mask, target_mask):
    # Pass the input sequence through the multi-head attention layer that attends to the target sequence
    self_attention_output = self.self_attention(inputs, inputs, inputs, target_mask)
    # Add the self-attention output to the input sequence
    self_attention_output = inputs + self.dropout(self_attention_output)
    # Pass the combined sequence through the multi-head attention layer that attends to the encoder outputs
    encoder_attention_output = self.encoder_attention(self_attention_output, encoded_input, encoded_input, input_mask)
    # Add the encoder-attention output to the combined sequence
    encoder_attention_output = self_attention_output + self.dropout(encoder_attention_output)
    # Pass the output sequence through the feed-forward layer
    feed_forward_output = self.feed_forward(encoder_attention_output)
    # Add the feed-forward output to the output sequence
    feed_forward_output = encoder_attention_output + self.dropout(feed_forward_output)
    # Pass the output sequence through the layer normalization layer
    output = self.layer_norm(feed_forward_output)
    return output
