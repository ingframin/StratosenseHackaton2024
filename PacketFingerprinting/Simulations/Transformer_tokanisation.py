import tensorflow as tf
import numpy as np

# ==== 🔹 Positional Encoding (Important for Sequential Data) 🔹 ====
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_encoding = self.get_positional_encoding(sequence_length, d_model)

    def get_positional_encoding(self, seq_len, d_model):
        positions = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(positions * div_term)
        pos_encoding[:, 1::2] = np.cos(positions * div_term)
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# ==== 🔹 Transformer Block (Multi-Head Attention + Feedforward) 🔹 ====
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# ==== 🔹 Full Transformer Encoder Model 🔹 ====
class TransformerEncoder(tf.keras.Model):
    def __init__(self, sequence_length, d_model, num_heads, dff, num_layers, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = tf.keras.layers.Dense(d_model)  # Project IQ data to `d_model` size
        self.pos_encoding = PositionalEncoding(sequence_length, d_model)
        self.encoder_layers = [TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(d_model, activation=None)  # Output embedding

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training)
        x = self.global_avg_pool(x)
        return self.output_layer(x)  # Final embedding (before L2 normalization)

# ==== 🔹 Custom Model with NT-Xent Contrastive Loss 🔹 ====
class CustomModel(tf.keras.Model):
    def __init__(self, sequence_length, d_model=128, num_heads=4, dff=256, num_layers=4, dropout_rate=0.1):
        super(CustomModel, self).__init__()
        self.encoder = TransformerEncoder(sequence_length, d_model, num_heads, dff, num_layers, dropout_rate)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def compile(self, optimizer, loss_fn):
        super(CustomModel, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        iq_batch = data  # Shape: (batch_size, sequence_length, 2) -> IQ data

        # Generate two augmentations (for NT-Xent Loss)
        iq_noisy = iq_batch + 0.01 * tf.random.normal(tf.shape(iq_batch))
        iq_shifted = tf.roll(iq_batch, shift=1, axis=1)

        with tf.GradientTape() as tape:
            z_i = self.encoder(iq_noisy, training=True)
            z_j = self.encoder(iq_shifted, training=True)
            loss = self.loss_fn(z_i, z_j)

        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def call(self, inputs):
        return self.encoder(inputs)

# ==== 🔹 NT-Xent Loss Function 🔹 ====
class NTXentLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def call(self, z_i, z_j):
        z_i = tf.math.l2_normalize(z_i, axis=1)
        z_j = tf.math.l2_normalize(z_j, axis=1)
        batch_size = tf.shape(z_i)[0]

        # Concatenate embeddings (batch-wise negatives)
        z = tf.concat([z_i, z_j], axis=0)
        similarity_matrix = tf.matmul(z, tf.transpose(z)) / self.temperature

        # Create labels (positive pairs are diagonals)
        labels = tf.eye(2 * batch_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels, similarity_matrix)
        
        return tf.reduce_mean(loss)

# ==== 🔹 Training Setup 🔹 ====
# Define model parameters
sequence_length = 1024  # Adjust to match IQ packet length
d_model = 128  # Size of embedding
num_heads = 4  # Multi-head attention heads
dff = 256  # Feedforward network size
num_layers = 4  # Number of Transformer blocks
dropout_rate = 0.1

# Instantiate model
model = CustomModel(sequence_length, d_model, num_heads, dff, num_layers, dropout_rate)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss_fn=NTXentLoss(temperature=0.1)
)

# Train model using TensorFlow's `fit()`
# Example dataset (random values for demonstration)
train_dataset = tf.data.Dataset.from_tensor_slices(
    tf.random.normal([10000, sequence_length, 2])
).batch(64)

model.fit(train_dataset, epochs=10)