import tensorflow as tf
import numpy as np
import os
import argparse
import ML_framework as mlf
import json 

"""
This file is used to generate data for the paper. 
The biggest update is using a flag for correctness instead of the correlation. 
We also include the recall score, as a metric as well as the precision. 

The threshold for the loss function will be based on increasing the recall. 

INPUT:  training data: duos of anchor and pos or neg 
OUTPUT: The score of similarity, 
"""



def main(output_shape, i):
    ## Parameters
    input_shape=512
    
    filepath= f'../Results/ADSB_Paper/ADSB_{output_shape}/Res_{i}/'

    ## Dataset 
    ds=mlf.DataHandler.DataHandlerMatthiasContast('../Dataset/Training/',input_shape=input_shape)

    # ## Model
    model_base= mlf.Model.Base(input_shape=input_shape,output_shape=output_shape,nr_layers=6,name='Model')
    model = model_base.get_model()

    ## Training 
    train_model(model,ds,filepath)
    ## Extracting th 
    th_recall,th_f1 = extact_th(model,ds,filepath)
    ## Testing
    test_model(model,ds,filepath,th_recall,th_f1)


def train_model(model,ds,filepath,epochs=60):
        
        if not os.path.exists(filepath+'Training'):
            os.makedirs(filepath+'Training')
            os.makedirs(filepath+'Results')
        fn_hist = filepath+"Training/training_history.json"
        filepath = filepath+'Training'+f'/model_weights_{model.name}'
        

        opt = tf.optimizers.Adam(learning_rate=0.0005)
        model.compile(optimizer=opt,weighted_metrics=[])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                    filepath=filepath,
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True,
                                    verbose=1, 
                                    restore_best_weights = True)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001, patience=15,mode='auto')
        history = model.fit(ds.train,validation_data=ds.val, epochs=epochs, 
                        callbacks=[es,checkpoint])
        # history = model.fit(ds.train,validation_data=ds.val, epochs=epochs, 
        #                 callbacks=[checkpoint])
        with open(fn_hist, "w") as f:
            json.dump(history.history, f,indent=4)
        model.load_weights(filepath)
        return history

def extact_th(model,ds,filepath=None):
    th = extact_th_save(model,ds,filepath)
    th_recall = extact_th_recall(model,ds,filepath=None)
    th_f1 = extact_th_F1(model,ds,filepath=None)
    return th_recall,th_f1

def extact_th_save(model,ds,filepath=None):
    loss,corr,flag= model.predict(ds.train)
    res = {'Loss':loss,'Corr':corr,'flag':flag}
    np.save(filepath+'Training/threshold',res)

    loss,corr,flag= model.predict(ds.val)
    res = {'Loss':loss,'Corr':corr,'flag':flag}
    np.save(filepath+'Training/threshold_val',res)
    

def cal_recall(loss,pos,th):
    org_pos = pos
    true_pos = (loss<th)&pos
    return np.sum(true_pos)/np.sum(org_pos)
def cal_precision(loss, pos,th):
    true_pos = (loss<th)&pos
    all_pos = (loss<th)
    return np.sum(true_pos)/np.sum(all_pos)
def F1_score(loss, pos,th):
    recall = cal_recall(loss,pos,th)
    precision = cal_precision(loss,pos,th)
    return 2*(recall*precision)/(recall+precision)

def extact_th_recall(model,ds,filepath=None):

    loss,corr,flag= model.predict(ds.train)
    # th = np.percentile(loss[flag==1],99) #go for 99%
    th = np.max([loss[flag==1]])

    print(f'\033[93mExtracted th recall:{th},recal:{cal_recall(loss,flag==1,th)},precision:{cal_precision(loss,flag==1,th)}\033[0m')
    loss,corr,flag= model.predict(ds.val)
    print(f'\033[93mValidationl,recal:{cal_recall(loss,flag==1,th)},precision:{cal_precision(loss,flag==1,th)}\033[0m')
    return th

def extact_th_F1(model,ds,filepath=None):

    loss,corr,flag= model.predict(ds.train)
    th_range = np.arange(0.0001,np.max(loss[flag==1]),0.00001)

    f1 = [F1_score(loss,flag==1,th) for th in th_range]
    th = th_range[np.argmax(f1)]

    print(f'\033[93mExtracted th recall:{th},recal:{cal_recall(loss,flag==1,th)},precision:{cal_precision(loss,flag==1,th)}\033[0m')
    loss,corr,flag= model.predict(ds.val)
    print(f'\033[93mValidationl,recal:{cal_recall(loss,flag==1,th)},precision:{cal_precision(loss,flag==1,th)}\033[0m')
    return th


def test_model(model, ds, filepath,th_recall,th_f1):
    loss,corr,flag= model.predict(ds.test())

    info = {
        'recall':{
            'precision': cal_precision(loss,flag==1,th_recall).astype(float),
            'recall': cal_recall(loss,flag==1,th_recall).astype(float),
            'th': th_recall.astype(float)
        },
        'F1':{
            'precision': cal_precision(loss,flag==1,th_f1).astype(float),
            'recall': cal_recall(loss,flag==1,th_f1).astype(float),
            'th': th_f1.astype(float)
        }

    } 

    with open(filepath+'Results/info.json', "w") as f:
        json.dump(info, f,indent=4)

    res = {'Loss':loss,'Corr':corr,'flag':flag}
    filepath = filepath+'Results/final'
    np.save(filepath,res)
    print(f'\033[93mTesting done:\033[0m')
    print(f'\033[93mF1: recall:{info["F1"]["recall"]*100:.2f}, precision:{info["F1"]["precision"]*100:.2f} \033[0m')
    print(f'\033[93mRecall: recall:{info["recall"]["recall"]*100:.2f}, precision:{info["recall"]["precision"]*100:.2f} \033[0m')




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

    task = [range(8),range(8,16),range(16,24),range(24,32)]
    sizes = [4,8,16,20,22,24,26,28,30,32,34,36,38,40,42,44,64]
    # # task = [[1,3,5,6,18,23,25],[7,9,10,11,19,22,26],[13,14,15,17,20,21,27]]
    for i in task[a.task]:
        for output_size in sizes: # [32,34,36,38,40]:
            main(output_size,i)
    # main(32,0)
    
