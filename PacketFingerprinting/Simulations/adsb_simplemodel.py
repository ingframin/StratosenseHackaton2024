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



def main(output_shape, i):
    ## Parameters
    input_shape=512
    
    filepath= f'../Results/ADSB_contrast2/ADSB_{output_shape}/Res_{i}/'

    ## Dataset 
    ds=mlf.DataHandler.DataHandlerMatthiasContast('../Dataset/Training/',input_shape=input_shape)

    # ## Model
    model_base= mlf.Model.Base(input_shape=input_shape,output_shape=output_shape,nr_layers=6,name='Model',Triple=False)
    model = model_base.get_model()

    ## Training 
    train_model(model,ds,filepath)
    ## Extracting th 
    th = extact_th(model,ds,filepath)
    ## Testing
    test_model(model,ds,filepath,th)


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
    # th = extact_th_highestacc(model,ds,filepath=None)
    th = extact_th_best_hit(model,ds,filepath=None)
    
    return th

def extact_th_save(model,ds,filepath=None):
    loss,corr= model.predict(ds.train)
    res = {'Loss':loss,'Corr':corr}
    np.save(filepath+'Training/threshold',res)

    loss,corr= model.predict(ds.val)
    res = {'Loss':loss,'Corr':corr}
    np.save(filepath+'Training/threshold_val',res)
    

def extact_th_best_hit(model,ds,filepath=None):
    ## Best detection
    loss,corr= model.predict(ds.train)
    
    def calc_acc(loss,corr,th):
        pos = loss[corr>0.90]
        neg = loss[corr<0.90]
        p = pos<th 
        n = neg<th
        pn=np.sum(p)/(np.sum(n)+np.sum(p))
        return pn
    
    th = np.percentile(loss[corr>0.90],99)
    acc_train = calc_acc(loss,corr,th)
    loss,corr= model.predict(ds.val)
    acc_val = calc_acc(loss,corr,th)
    print(f'\033[93mExtracted th:{th}. acc_train:{acc_train},acc_val:{acc_val}\033[0m')
    return th

def extact_th_highestacc(model,ds,filepath=None):
    ## Best detection
    loss,corr= model.predict(ds.train)
    def calc_acc(loss,corr,th):
        pos = loss[corr>0.90]
        neg = loss[corr<0.90]
        p = pos<th 
        n = neg>th
        pn=(np.sum(n)+np.sum(p))/(len(n)+len(p))
        return pn
    
    th_range = np.arange(0,1,0.00001)
    res = []

    for th in th_range:
        res.append(calc_acc(loss,corr,th))
    
    th = th_range[np.argmax(np.array(res))]
    acc_train = calc_acc(loss,corr,th)
    loss,corr= model.predict(ds.val)
    acc_val = calc_acc(loss,corr,th)

    print(f'\033[93mExtracted th:{th}. acc_train:{acc_train},acc_val:{acc_val}\033[0m')
    return th



def test_model(model, ds, filepath,th):
    p,n= model.predict(ds.test())
    def calc_acc(loss,corr,th):
        pos = loss[corr>0.90]
        neg = loss[corr<0.90]
        p = pos<th 
        n = neg<th
        pn=np.sum(p)/(np.sum(n)+np.sum(p))
        return pn
    acc = calc_acc(p,n,th)
    res = {'Loss':p,'Corr':n}
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

    task = [[0,3,6],[1,4,7],[2,5]]
    task = [[8,9,10],[11,12,13],[14,15,16]]
    sizes = [4,8,16,20,22,24,26,28,30,32]
    # task = [[1,3,5,6,18,23,25],[7,9,10,11,19,22,26],[13,14,15,17,20,21,27]]
    for i in task[a.task]:
        for output_size in sizes: # [32,34,36,38,40]:
            main(output_size,i)
    
