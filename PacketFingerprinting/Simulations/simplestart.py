import tensorflow as tf
import numpy as np
import os
import argparse
import ML_framework as mlf
import json 

"""
This file is a simple start to test out models without overloading the jupiter notebook. 

INPUT:  training data: Triplets with (anchor, positive and negative)
        Validation and test data: (anchor,test)
OUTPUT: The score of similarity, 
"""


class DataHandler:
    def __init__(self,inputsize) -> None:
        self.inputsize=inputsize
        self.batchsize=2
        
    def get_dataset_train(self,nr):
        data= np.random.random(size=(nr,self.inputsize,2))
        ds  = tf.data.Dataset.from_tensor_slices((data,data,data))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)
    
    def get_dataset_eval(self,nr):
        data= np.random.random(size=(nr,self.inputsize,2))
        ds  = tf.data.Dataset.from_tensor_slices((data,data))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)


def main():

    ## Dataset 
    inputsize=128
    ds=DataHandler(inputsize)

    ## Model
    model_base= mlf.Model.Base(input_shape=inputsize,output_shape=32,nr_layers=4,name='Test')
    model = model_base.get_model()
 
    ## Training 
    model.compile(optimizer='adam')
    model.fit(ds.get_dataset_train(10),epochs=8,validation_data=ds.get_dataset_eval(10))

    ## Extracting th 
    ## Testing


def test_data():
    # process the inputs
    # calculate loss 
    # decide based on loss 
    # build accuracy
    pass




if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Script so useful.')
    p.add_argument("--gpu", type=int, default=7)
    a = p.parse_args()
    print(f'\033[93mRunning Complexities on GPU {a.gpu} \033[0m')

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{a.gpu}'
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    np.random.seed(2323)
    tf.random.set_seed(2323)

    main()
