import tensorflow as tf
import numpy as np

""" Models"""
   # cnns=4
    # connected=(128,128)
    # output_shape = 6

    # input_shape= inputsize
    # input_layer = tf.keras.Input(shape=(input_shape,2))
    # x = tf.keras.layers.Reshape((1,input_shape,2))(input_layer)
    # x =  tf.keras.layers.Conv2D(filters=32,kernel_size=(1,1))(x)
    # for _ in range(cnns):
    #     x =  tf.keras.layers.Conv2D(filters=32,kernel_size=(1,3))(x)
    #     x = tf.keras.layers.MaxPool2D((1,2))(x)

    # for nn in connected:
    #     x = tf.keras.layers.Dense(nn,activation=tf.keras.activations.selu)(x)
    #     x= tf.keras.layers.Dropout(0.1)(x)
    # out = tf.keras.layers.Dense(output_shape,activation=tf.keras.activations.softmax, name="Exit")(x)

    # Smodel = CModel(input_layer,out)

class EvaluateModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.margin=0.7
        self.th=None
    
    @tf.function
    def predict_step(self,inputs):
        anchor_embed = self(inputs[0],training=False)
        test_embed = self(inputs[1],training=False)
        corr, ia,it,flag = tf.unstack(inputs[2], axis=-1)
        y_true = tf.cast(corr,tf.float32)
        distances =tf.sqrt(tf.reduce_sum(tf.square(anchor_embed - test_embed), axis=1))
        return tf.square(distances), y_true, anchor_embed,test_embed ,ia,it,flag## give back loss + correlation factor + raw_results

    @property
    def metrics(self):
        return [self.loss_tracker]



class ContrastModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.margin=0.7
        self.th=None
    
    @tf.function
    def predict_step(self,inputs):
        anchor_embed = self(inputs[0],training=False)
        test_embed = self(inputs[1],training=False)
        corr,ham,flag = tf.unstack(inputs[2], axis=-1)
        distances =tf.sqrt(tf.reduce_sum(tf.square(anchor_embed - test_embed), axis=1))
        return anchor_embed,test_embed, tf.square(distances), corr,flag, ham ## give back loss + correlation factor + raw_results

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            # calculate the loss of the siamese network
            loss = self._compute_loss(inputs)
        # compute the gradients and optimize the model
        gradients = tape.gradient(
            loss,
            self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        self.loss_tracker.update_state(loss)
        return  {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, inputs):
        loss = self._compute_loss(inputs)
        self.loss_tracker.update_state(loss)
        return  {m.name: m.result() for m in self.metrics}

    def _compute_loss(self, inputs,training=True):
        anchor_embed = self(inputs[0],training=training)
        test_embed = self(inputs[1],training=training)
        corr,ham,flag = tf.unstack(inputs[2], axis=-1)
        corr = tf.cast(flag,tf.float32)

        distances =tf.sqrt(tf.reduce_sum(tf.square(anchor_embed - test_embed), axis=1))

        # Contrastive loss calculation
        positive_loss = corr * tf.square( distances ) # For similar pairs (y=1)
        negative_loss = (1-corr) * tf.square( tf.maximum(self.margin - distances, 0))  # For dissimilar pairs (y=0)
        
        # Combine and compute the mean loss
        loss = tf.reduce_mean(positive_loss + negative_loss)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker]

class TModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_tracker_p = tf.keras.metrics.Mean(name="loss_p")
        self.loss_tracker_n = tf.keras.metrics.Mean(name="loss_n")
        self.margin=0.2
        self.th=None
    
    @tf.function
    def predict_step(self,inputs):
        (apDistance, anDistance) = self._compute_distance(inputs,training=False)
        return (apDistance, anDistance)

    def extract_th(self,ds):
        ## Implement this when there is a better view on the data
        pass

    @tf.function
    def train_step(self, inputs):
        # print(inputs)
        with tf.GradientTape() as tape:
            (apDistance, anDistance) = self._compute_distance(inputs)
            # calculate the loss of the siamese network
            loss = self._compute_loss(apDistance, anDistance)
        # compute the gradients and optimize the model
        gradients = tape.gradient(
            loss,
            self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables)
        )

        self.loss_tracker.update_state(loss)
        self.loss_tracker_p.update_state(apDistance)
        self.loss_tracker_n.update_state(anDistance)

        return  {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, inputs):
        # print(inputs)
        
        (apDistance, anDistance) = self._compute_distance(inputs,training=False)
        # calculate the loss of the siamese network
        loss = self._compute_loss(apDistance, anDistance)

        self.loss_tracker.update_state(loss)
        self.loss_tracker_p.update_state(apDistance)
        self.loss_tracker_n.update_state(anDistance)

        return  {m.name: m.result() for m in self.metrics}

    def _compute_loss(self, apDistance, anDistance):
        loss = apDistance - anDistance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def _compute_distance(self, inputs,training=True):
        embeddings = [self(ii,training=training) for ii in inputs]

        # calculate the anchor to positive and negative distance
        apDistance = tf.reduce_sum(
            tf.square(embeddings[0] - embeddings[1]), axis=-1
        )
        anDistance = tf.reduce_sum(
            tf.square(embeddings[0] - embeddings[2]), axis=-1
        )
        
        # return the distances
        return (apDistance, anDistance)
    
    @property
    def metrics(self):
        return [self.loss_tracker,self.loss_tracker_p,self.loss_tracker_n]    

class Model_Base:
    def __init__(self,name) -> None:
        self.name= name
        pass
    def build(self,m=None):
        pass
    def get_model(self,m=None):
        pass


class Base(Model_Base):
    """ The basic model oshea resnet """
    def __init__(self,input_shape=512,output_shape=6,nr_layers=4,name='Base',Triple=False) -> None:
        super().__init__(name=name)
        self.out_model=TModel if Triple else ContrastModel
        
        ## Inputs
        self.input_layer = tf.keras.Input(shape=(input_shape,2))
        self.reshaper = tf.keras.layers.Reshape((1,input_shape,2))
        self.featEx = FeatureExtraction(nr_layers=nr_layers)
        self.output= Exit(output_shape)
    
    def build(self):
        x = self.reshaper(self.input_layer)
        x = self.featEx(x)
        out = self.output(x)
        return out, 'Full_model'
         
    def get_model(self):
        out,post = self.build()
        return self.out_model(self.input_layer,out,name=post) 


"""Blocks"""

class FeatureExtraction(tf.keras.Model):
    def __init__(self, nr_layers=7, filters=32,kernel_size=(1,3)):
        super(FeatureExtraction, self).__init__()
        self.res_stacks = [ResidualStack(filters,kernel_size) for _ in range(nr_layers) ]

    def call(self, x, training=False):
        for stack in self.res_stacks:
            x = stack(x,training=training)
        return x

class CNNStack(tf.keras.Model):
    """ This is a resnet unit block described by Oshea"""
    def __init__(self, filters=32,kernel_size=(1,3)):
        super(ResidualStack, self).__init__()
        self.conv_relu_1 = tf.keras.layers.Conv2D(filters=filters,kernel_size=(1,1),activation=tf.keras.activations.relu, padding='same')
        self.conv_linear = tf.keras.layers.Conv2D(filters=filters,kernel_size=(1,1),activation=tf.keras.activations.linear, padding='same')
        self.max_pool = tf.keras.layers.MaxPool2D((1,2))

    def call(self, input_tensor, training=False):
        x = self.conv_linear(input_tensor,training=training)
        x = self.res_unit_1(x, training=training)
        x = self.res_unit_2(x, training=training)
        out = self.max_pool(x)
        return out


class ResidualStack(tf.keras.Model):
    """ This is a resnet unit block described by Oshea"""
    def __init__(self, filters=32,kernel_size=(1,3)):
        super(ResidualStack, self).__init__()
        self.conv_linear = tf.keras.layers.Conv2D(filters=filters,kernel_size=(1,1),activation=tf.keras.activations.linear, padding='same')
        self.res_unit_1 = ResidualUnit(filters=filters,kernel_size=kernel_size)
        self.res_unit_2 = ResidualUnit(filters=filters,kernel_size=kernel_size)
        self.max_pool = tf.keras.layers.MaxPool2D((1,2))

    def call(self, input_tensor, training=False):
        x = self.conv_linear(input_tensor,training=training)
        x = self.res_unit_1(x, training=training)
        x = self.res_unit_2(x, training=training)
        out = self.max_pool(x)
        return out
  
class ResidualUnit(tf.keras.Model):
    """ This is a residual unit block described by Oshea"""
    def __init__(self, filters=32,kernel_size=(1,3)):
        super(ResidualUnit, self).__init__()
        self.conv_relu = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,activation=tf.keras.activations.relu, padding='same')
        self.conv_linear = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,activation=tf.keras.activations.linear, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.bn2 =  tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        fx = self.conv_relu(input_tensor,training=training)
        fx = self.bn(fx)
        fx = self.conv_linear(fx,training=training)
        out = input_tensor+fx ## skip connection 
        out = self.relu(out)
        out = self.bn2(out)
        return out
    
class Exit(tf.keras.Model):
    """ Decision layer, this are multiple fully conected stacks + classefing stack """
    def __init__(self,output_shape,N_per_dense = (512,512) ):
        super(Exit, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_layers = [tf.keras.layers.Dense(nn,activation=tf.keras.activations.selu) for nn in N_per_dense ]
        self.dropout = tf.keras.layers.Dropout(0.1)
        # self.output_layer = tf.keras.layers.Dense(output_shape,activation=tf.keras.activations.softmax, name="Exit")
        self.output_layer = tf.keras.layers.Dense(output_shape,activation=None, name="Exit")

    def call(self, input_tensor, training=False):
        x = self.flatten(input_tensor)
        for layer in self.dense_layers:
            x = layer(x,training=training)
            x = self.dropout(x,training=training)
        out = self.output_layer(x,training=training)
        return out

    def get_output(self):
        return self.output_layer
    

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
class TransformerEncoder(Model_Base):
    def __init__(self, sequence_length, d_model, num_heads, dff, num_layers, dropout_rate=0.1, name='transfo'):
        super().__init__(name=name)
        self.input_layer = tf.keras.Input(shape=(sequence_length,2))
        self.embedding = tf.keras.layers.Dense(d_model)  # Project IQ data to `d_model` size
        self.pos_encoding = PositionalEncoding(sequence_length, d_model)
        self.encoder_layers = [TransformerBlock(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.output_layer = tf.keras.layers.Dense(d_model, activation=None)  # Output embedding

    def build(self):
        x = self.embedding(self.input_layer)
        x = self.pos_encoding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = self.global_avg_pool(x)
        out = self.output_layer(x) 
        return out, 'Tranformer'  # Final embedding (before L2 normalization)
    
    def get_model(self):
        out,post = self.build()
        return ContrastModel(self.input_layer,out,name=post) 
    

