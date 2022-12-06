class AttentionLayer(tf.keras.layers.Layer):
  def __init__(self, attention_mechanism):
    super(AttentionLayer, self).__init__()
    self.attention_mechanism = attention_mechanism

  def call(self, inputs):
    input_sequence, output_sequence = inputs
    attention_weights = self.attention_mechanism(input_sequence, output_sequence)
    weighted_input = attention_weights * input_sequence
    weighted_input = tf.reduce_sum(weighted_input, axis=-1)
    return weighted_input
