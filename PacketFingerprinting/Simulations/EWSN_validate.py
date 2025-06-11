import tensorflow as tf
import numpy as np
import os
import argparse
import ML_framework as mlf
import json 
import tensorflow.keras.backend as K
import pandas as pd
"""
This file is used to Process the data for EWSN. 
As we skrewed up, we have to reprocess the dataset. 
This script will get the embedding for all the test data. 

"""



def main(output_shape, file_base='../Results/EWSN/', model_name='ResNet', task=0, input_size=64):
    ## Parameters
    input_shape=input_size
    layers = {16:2,32:3,64:4,128:5,256:6}

    ## Dataset
    # ds=mlf.DataHandler.DataHandlerEWSN_Evaluation('../Dataset/EWSN_disjunct/',input_shape=input_shape,batchsize=64, uid='eval')
    ds=mlf.DataHandler.DataHandlerEWSN_Evaluation('../Dataset/EWSN_disjunct/',input_shape=input_shape,batchsize=64, uid='val')
    
    model_base= mlf.Model.Base(input_shape=input_shape,output_shape=output_shape,nr_layers=layers[input_size],name='Model')
    
    model = model_base.get_model()

    ## training
    filepath = file_base+f'{model_name}_{output_shape}/res_{task}/'
    fp_checkpoints = filepath+'Training'+f'/model_weights_{model.name}'
    model.load_weights(fp_checkpoints).expect_partial()

    ## Save results for threshold determination 
    embeddings,_, _, _,_, idx = model.predict(ds.test())
    res = {'idx':idx,'embedding':embeddings}
    np.save(filepath+'Results/embeddings_val',res)




if __name__ == "__main__":

    p = argparse.ArgumentParser(description='Script so useful.')
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--size", type=int, default=64)
    a = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{a.gpu}'
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    np.random.seed(2323)
    tf.random.set_seed(2323)

    sizes = [4,6,8,10,12,14,16,18,20,22,24]
    tasks = range(4)
    inputs = (16,32,64,128,256)

    
    for isize in inputs:
        for size in sizes:
            for task in tasks:
                fn=f'../Results/EWSN_ds_{isize}/'
                print(f'\033[93mRunning Resnet {isize} {size} {task} on GPU {a.gpu} \033[0m')
                main(output_shape=size,model_name='ResNet', task=task, file_base=fn, input_size=isize)
        



