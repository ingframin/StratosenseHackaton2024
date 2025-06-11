import tensorflow as tf
import numpy as np
import ML_framework as mlf
import os
import argparse

import json 


def main():
    sizes = [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
    input_shape=256
    result = {}


    ## Transformer
    ## model 
    
    num_heads = 4  # Multi-head attention heads
    dff = 256  # Feedforward network size
    num_layers = 4  # Number of Transformer blocks
    dropout_rate = 0.1
    for model_name in ['ResNet']:#['Transformer','ResNet']: 
        
        for input_shape, l in [(16,2),(32,3),(64,4),(128,5),(256,6)]:
            res_temp ={}
            for output_shape in sizes:
                model_base= mlf.Model.Base(input_shape=input_shape,output_shape=output_shape,nr_layers=l,name='Model')
        
                model = model_base.get_model()
                flops,params = get_mflops(model)
                res_temp[output_shape] = {model.name:{'Mflops':flops,'params':params}}
                print(model_name, input_shape, output_shape, flops, params)
            result[f'{model_name}_{input_shape}'] = res_temp

 
    with open(f'../Results/complexity_2.json', "w") as outfile:
            json.dump(result, outfile,indent=4)
    



def get_mflops(model):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

    # Compute FLOPs for one sample
    batch_size = 1
    inputs = [ tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs]

    # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPs with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (
        tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
        )
        .with_empty_output()
        .build()
    )

    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )
    return float((flops.total_float_ops / 1e6)) / 2,float(np.sum([np.prod(v.get_shape()) for v in model.trainable_weights]))


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

    
