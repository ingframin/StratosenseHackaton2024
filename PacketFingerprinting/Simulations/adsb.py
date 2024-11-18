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
    
    filepath= f'../Results/ADSB/ADSB_{output_shape}/Res_{i}/'

    ## Dataset 
    ds=mlf.DataHandler.DataHandlerMatthias('../Dataset/Training/',input_shape=input_shape)
    ## Model
    model_base= mlf.Model.Base(input_shape=input_shape,output_shape=output_shape,nr_layers=6,name='Model')
    model = model_base.get_model()
    ## Training 
    train_model(model,ds,filepath)
    ## Extracting th 
    th = extact_th(model,ds,filepath)
    ## Testing
    test_model(model,ds,filepath,th)


def train_model(model,ds,filepath,epochs=100):
        
        if not os.path.exists(filepath+'Training'):
            os.makedirs(filepath+'Training')
            os.makedirs(filepath+'Results')
        fn_hist = filepath+"Training/training_history.json"
        filepath = filepath+'Training'+f'/model_weights_{model.name}'
        

        opt = tf.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt,weighted_metrics=[])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                    filepath=filepath,
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True,
                                    verbose=1, 
                                    restore_best_weights = True)
        # es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001, patience=3,mode='auto')
        # history = model.fit(ds.train,validation_data=ds.val, epochs=epochs, 
        #                 callbacks=[es,checkpoint])
        history = model.fit(ds.train,validation_data=ds.val, epochs=epochs, 
                        callbacks=[checkpoint])
        with open(fn_hist, "w") as f:
            json.dump(history.history, f,indent=4)
        model.load_weights(filepath)
        return history

def extact_th(model,ds,filepath=None):
    p,n= model.predict(ds.train)
    # th_range = np.arange(0,2,0.001)
    # # select th for which the accuracy of the prediction is maximum.
    def calc_acc(pd,nd,th):
        p = pd<th
        n = nd>th
        return (np.sum(n)+np.sum(p))/(len(n)+len(p))
    # res=[]
    # for th in th_range:
    #     res.append(calc_acc(p,n,th))
    # res = np.array(res)
    
    # th = th_range[np.argmax(res)]
    th=np.percentile(p,99)
    acc_train = calc_acc(p,n,th)
    p,n= model.predict(ds.val)
    acc_val = calc_acc(p,n,th)
    print(f'\033[93mExtracted th:{th}. acc_train:{acc_train},acc_val:{acc_val}\033[0m')
    return th



def test_model(model, ds, filepath,th):
    p,n= model.predict(ds.test())
    def calc_acc(pd,nd,th):
        p = pd<th
        n = nd>th
        return (np.sum(n)+np.sum(p))/(len(n)+len(p))
    acc = calc_acc(p,n,th)
    res = {'POS':p,'NEG':n}
    with open(filepath+'Results/info', "w") as f:
        json.dump({'acc':acc}, f,indent=4)
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

    task = [[2,4,8,12,16],[24,32,48],[64,96,128]]
    task = [[1,3,5,6,18,23,25],[7,9,10,11,19,22,26],[13,14,15,17,20,21,27]]
    # for output_shape in task[a.task]:
    #     for i in range(8):
    main(128,2)
