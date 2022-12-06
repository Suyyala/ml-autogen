class Encoder(keras.Model):
  def __init__(self, num_heads, input_dim, output_dim):
    super(Encoder, self).__init__()
    self.num_heads = num_heads
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.attention_mechanism = MultiHeadAttention(num_heads=num_heads,
                                                  input_dim=input_dim,
                                                  output_dim=output_dim,
                                                  return_attention=False)
    self.feed_forward_network = FeedForward(input_dim=input_dim,
                                            hidden_dim=input_dim,
                                            output_dim=output_dim,
                                            activation="relu")
    self.output_normalization = tf.keras.layers.LayerNormalization()
    self.positional_encoding = tf.Variable(
        self.positional_encoding_init(input_dim, output_dim),
        name="positional_encoding",
        dtype=tf.float32,
        trainable=False)

  @staticmethod
  def positional_encoding_init(input_dim, output_dim):
    # create a positional encoding matrix with shape [input_dim, output_dim]
    # using sine and cosine functions of different frequencies

  def call(self, inputs):
    x = self.attention_mechanism(inputs)
    x = self.feed_forward_network(x)
    x = self.output_normalization(x + inputs)
    x += self.positional_encoding
    return x

  
class Decoder(keras.Model):
  def __init__(self, num_heads, input_dim, output_dim):
    super(Decoder, self).__init__()
    self.num_heads = num_heads
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.attention_mechanism = MultiHeadAttention(num_heads=num_heads,
                                                  input_dim=input_dim,
                                                  output_dim=output_dim,
                                                  return_attention=False)
    self.encoder_decoder_attention_mechanism = MultiHeadAttention(
        num_heads=num_heads,
        input_dim=input_dim,
        output_dim=output_dim,
        return_attention=False,
        causality=True)
    self.feed_forward_network = FeedForward(input_dim=input_dim,
                                            hidden_dim=input_dim,
                                            output_dim=output_dim,
                                            activation="relu")
    self.output_normalization = tf.keras.layers.LayerNormalization()

  def call(self, inputs, encoder_outputs):
    x = self.attention_mechanism(inputs)
    x = self.encoder_decoder_attention_mechanism(x, encoder_outputs)
    x = self.
