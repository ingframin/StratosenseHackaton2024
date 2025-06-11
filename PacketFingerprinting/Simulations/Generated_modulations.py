import tensorflow as tf
import numpy as np
import os
import argparse
import ML_framework as mlf
import json 
import tensorflow.keras.backend as K

"""
This file is used to Process the data for EWSN. 

We generated a dataset by capturing and decoding adsb messages. 

We have to do: 
    1. Generate dataset interpreter
    2. Make transformer model 
    3. Grid search  

Mattias has given us a dataset of triplets. We make one dataset with 1/3 matches, 2/3 non matching. 
We will use contrastive (siamese) loss using eqlueian space. 

We will need to find a threhold to see if samples are similar or not (classification quality?)



INPUT:  training data: duos of anchor and pos or neg 
OUTPUT: The score of similarity, 
"""



def main(output_shape, file_base='../Results/AMC/', model_name='ResNet', task=0,id='n'):
    ## Parameters
    input_shape=256

    ## Dataset
    # ds=mlf.DataHandler.DataHandlerEWSN2('../Dataset/EWSN/',input_shape=input_shape,batchsize=32)
    ds=mlf.DataHandler.DataHandlerAMC('../Dataset/AMC/',input_shape=input_shape,batchsize=64,id=id)

    # ## model 
    
    num_heads = 4  # Multi-head attention heads
    dff = 256  # Feedforward network size
    num_layers = 4  # Number of Transformer blocks
    dropout_rate = 0.1

    if model_name=="Transformer":
        model_base = mlf.Model.TransformerEncoder(input_shape,output_shape, num_heads, dff, num_layers, dropout_rate,name='tranfo')

    

    # # ## Model #0.0135!
    if model_name=="ResNet":
        model_base= mlf.Model.Base(input_shape=input_shape,output_shape=output_shape,nr_layers=6,name='Model')
    
    model = model_base.get_model()

    ## training
    epochs=40
    filepath = file_base+f'/{model_name}_{output_shape}/res_{task}/'
    if not os.path.exists(filepath+'Training'):
        os.makedirs(filepath+'Training')
        os.makedirs(filepath+'Results')
    fn_hist = filepath+"Training/training_history.json"
    fp_checkpoints = filepath+'Training'+f'/model_weights_{model.name}'
        

    opt = tf.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt,weighted_metrics=[])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                filepath=fp_checkpoints,
                                save_weights_only=True,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True,
                                verbose=1, 
                                restore_best_weights = True)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.001, patience=10,mode='auto')
    history = model.fit(ds.train,validation_data=ds.val, epochs=epochs, 
                    callbacks=[es,checkpoint])

    with open(fn_hist, "w") as f:
        json.dump(history.history, f,indent=4)
    model.load_weights(fp_checkpoints)


        ## Save results for threshold determination 
    info = {}

    anchors,tests, loss, mod_i,flag, mod_j = model.predict(ds.train)
    res = {'A':anchors,'T':tests, 'Loss':loss,'mod_i':mod_i,'flag':flag,'mod_j':mod_j}
    np.save(filepath+'Results/train',res)
    info['Train'] = {'loss':float(np.mean(loss)),'loss_pos':float(np.mean(loss[flag==1])),'loss_neg':float(np.mean(loss[flag==0]))}

    anchors,tests, loss, mod_i,flag, mod_j = model.predict(ds.val)
    res = {'A':anchors,'T':tests, 'Loss':loss,'mod_i':mod_i,'flag':flag,'mod_j':mod_j}
    np.save(filepath+'Results/val',res)
    info['Val'] = {'loss':float(np.mean(loss)),'loss_pos':float(np.mean(loss[flag==1])),'loss_neg':float(np.mean(loss[flag==0]))}

    anchors,tests, loss, mod_i,flag, mod_j = model.predict(ds.test())
    res = {'A':anchors,'T':tests, 'Loss':loss,'mod_i':mod_i,'flag':flag,'mod_j':mod_j}
    np.save(filepath+'Results/test',res)
    info['Test'] = {'loss':float(np.mean(loss)),'loss_pos':float(np.mean(loss[flag==1])),'loss_neg':float(np.mean(loss[flag==0]))}
    print(info)

    with open(filepath+'Results/info.json', "w") as f:
        json.dump(info, f,indent=4)





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

    sizes = [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]

    for size in sizes:
        for fn,id in zip(['../Results/AMC/','../Results/AMC_shift/'],['n','shift']):
            K.clear_session()
            print(f'\033[93mRunning Transformer {size} on GPU {a.gpu} \033[0m')
            main(output_shape=size,model_name='Transformer', task=a.task,file_base=fn,id=id)
            print(f'\033[93mRunning Resnet {size} on GPU {a.gpu} \033[0m')
            main(output_shape=size,model_name='ResNet', task=a.task,file_base=fn,id=id)
        



