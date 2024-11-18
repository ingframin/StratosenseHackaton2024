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



class CModel(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.margin=0.7
        self.th=None
    
    @tf.function
    def predict_step(self,inputs):
        anchor_embed = self(inputs[0],training=False)
        test_embed = self(inputs[1],training=False)
        y_true = tf.cast(inputs[2],tf.float32)

        distances =tf.sqrt(tf.reduce_sum(tf.square(anchor_embed - test_embed), axis=1))

        return tf.square(distances), y_true ## give back loss + correlation factor + raw_results

    def extract_th(self,ds):
        ## Implement this when there is a better view on the data
        pass

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
    

    def _compute_loss_predict(self, inputs,training=True):
        anchor_embed = self(inputs[0],training=training)
        test_embed = self(inputs[1],training=training)
        y_true = tf.cast(inputs[2],tf.float32)

        distances =tf.sqrt(tf.reduce_sum(tf.square(anchor_embed - test_embed), axis=1))

        return tf.square(distances), y_true, anchor_embed,test_embed

    @tf.function
    def test_step(self, inputs):
        loss = self._compute_loss(inputs)
        self.loss_tracker.update_state(loss)
        return  {m.name: m.result() for m in self.metrics}

    def _compute_loss(self, inputs,training=True):
        anchor_embed = self(inputs[0],training=training)
        test_embed = self(inputs[1],training=training)
        y_true = tf.cast(inputs[2],tf.float32)

        distances =tf.sqrt(tf.reduce_sum(tf.square(anchor_embed - test_embed), axis=1))

        # Contrastive loss calculation
        positive_loss = y_true * tf.square( distances ) # For similar pairs (y=1)
        negative_loss = (1-y_true) * tf.square( tf.maximum(self.margin - distances, 0))  # For dissimilar pairs (y=0)
        
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

class BasicCNN(Model_Base):
    """ The basic model oshea resnet """
    def __init__(self,input_shape=512,output_shape=6,nr_layers=4,name='Base') -> None:
        super().__init__(name=name)
        
        ## Inputs
        self.input_layer = tf.keras.Input(shape=(input_shape,2))
        self.reshaper = tf.keras.layers.Reshape((1,input_shape,2))
        self.featEx = FeatureExtraction(nr_layers=nr_layers)
        self.output= Exit(output_shape)
    
    def build(self):
        x = self.reshaper(self.input_layer)
        x= self.featEx(x)
        out = self.output(x)
        return out, 'Full_model'
         
    def get_model(self):
        out,post = self.build()
        return CModel(self.input_layer,out,name=post)

class Base(Model_Base):
    """ The basic model oshea resnet """
    def __init__(self,input_shape=512,output_shape=6,nr_layers=4,name='Base',Triple=True) -> None:
        super().__init__(name=name)
        self.out_model=TModel if Triple else CModel
        
        ## Inputs
        self.input_layer = tf.keras.Input(shape=(input_shape,2))
        self.reshaper = tf.keras.layers.Reshape((1,input_shape,2))
        self.featEx = FeatureExtraction(nr_layers=nr_layers)
        self.output= Exit(output_shape)
    
    def build(self):
        x = self.reshaper(self.input_layer)
        x= self.featEx(x)
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
        self.output_layer = tf.keras.layers.Dense(output_shape,activation=tf.keras.activations.softmax, name="Exit")

    def call(self, input_tensor, training=False):
        x = self.flatten(input_tensor)
        for layer in self.dense_layers:
            x = layer(x,training=training)
            x = self.dropout(x,training=training)
        out = self.output_layer(x,training=training)
        return out

    def get_output(self):
        return self.output_layer
