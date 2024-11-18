import tensorflow as tf
import numpy as np
import os
import argparse
import ML_framework as mlf
import json 

"""
With this file we will evaluate our model and try to further compress the features based on a completele diffetent data set.


"""



def main(output_shape, i):
    ## Parameters
    input_shape=512
    fn_model_base = f'../Results/ADSB_contrast2/ADSB_{output_shape}/Res_{i}/'
    
    filepath= f'../Results/ADSB_Validation2/ADSB_{output_shape}_big/'

    ## Dataset 
    ds=mlf.DataHandler.DataHandlerMatthiasEvaluate2('../Dataset/Validation/',input_shape=input_shape,batchsize=2048)


    # # ## Model
    model_base= mlf.Model.Base(input_shape=input_shape,output_shape=output_shape,nr_layers=6,name='Model',Triple=False)
    model_base.out_model= mlf.Model.EvaluateModel
    model = model_base.get_model()

    ## Load model 
    #../Results/ADSB_contrast2/ADSB_4/Res_0/Training/model_weights_Full_model
    model.load_weights(fn_model_base+'Training/model_weights_Full_model').expect_partial()

    with open(fn_model_base+'Results/info', "r") as outfile:
        info =  json.load(outfile)
        th=info['th']
    
    ## Testing
    test_model(model,ds,filepath,th)


def test_model(model, ds, filepath,th):
    if not os.path.exists(filepath+'Results'):
        os.makedirs(filepath+'Results')

    p,n,a,t,ia,it,flag= model.predict(ds.test())
    def calc_acc(loss,corr,th):
        pos = loss[corr>0.90]
        neg = loss[corr<0.90]
        p = pos<th 
        n = neg<th
        pn=np.sum(p)/(np.sum(n)+np.sum(p))
        return pn
    acc = calc_acc(p,n,th)
    res = {'Loss':p,'Corr':n,'anchor':a,'test':t,'idx_a':ia,'idx_t':it,'f':flag}
    with open(filepath+'Results/info', "w") as f:
        json.dump({'acc':acc,'th':th}, f,indent=4)
    filepath = filepath+'Results/final'
    np.save(filepath,res)
    print(f'\033[93mTesting done: accuracy {acc*100:.2f}\033[0m')




if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Script so useful.')
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--task", type=int, default=0)
    a = p.parse_args()
    print(f'\033[93mRunning Complexities on GPU {a.gpu} \033[0m')

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{a.gpu}'
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    np.random.seed(2323)
    tf.random.set_seed(2323)
    bestest = [(4, 6), (8, 0), (16, 15), (20, 6), (22, 14), (24, 11), (26, 7), (28, 14), (30, 1), (32, 6),(34, 3), (36, 2), (38, 6), (40, 7)]
    # task = [[0,3,6],[1,4,7],[2,5]]
    bestest = [(24, 11), (26, 7), (28, 14), (30, 1), (32, 6),]
    for output_size,i in bestest:
        main(output_size,i)
    # main(32,2)
    
