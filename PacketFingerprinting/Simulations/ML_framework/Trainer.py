import tensorflow as tf
import numpy as np
import os
# from .DataHandler import DataHandler_Base
# from .DataHandler import DataHandler_ORAN, DataHandler_Single
from .Model import Model_Base
from .Model import Base
import json


class Trainer_Base:
    """ The trainer, trains and tests the different models and is in charge of correctly storing the results
        It trains the models with the correct dataset and the correct indexes, assuring reproduceblity"""
    def __init__(self,id:str,dataset:DataHandler_Base,model:Model_Base) -> None:
        self.id=id
        self.dataset= dataset
        self.model_base = model
        self.base_dir = f'../Results/{self.id}/{self.model_base.name}/'
        self.info = {'Training':{},'Results':{}}
        

        ## make sure folders exists
        if not os.path.exists(self.base_dir+'Training'):
            os.makedirs(self.base_dir+'Training') 

        if not os.path.exists(self.base_dir+'Results'):
            os.makedirs(self.base_dir+'Results') 
        
        ## load idx if exists 
        fn = f'../Results/{self.id}/idx'
        if not os.path.exists(fn+'.npz'):
            self.dataset.idx.save(fn)
        self.dataset.idx.load(fn+'.npz')
    
    def save_info(self):
        with open(f'{self.base_dir}info.json', "w") as outfile:
            json.dump(self.info, outfile,indent=4)
    
    def load_info(self,fn=None):
        if fn is None:
            fn = self.base_dir
        with open(fn+'info.json', "r") as outfile:
            self.info = json.load(outfile)
    
    def train(self, learning_rate=0.00001):
        model = self.model_base.get_model()
        self.train_model(model,self.dataset.idx.train,self.dataset.idx.val,learning_rate=learning_rate)
        self.save_info()
        self.test_model(model)
        self.save_info()
    
    def test(self,fn=None):
        self.load_info(fn)
        self.load_model(fn)
        model = self.model_base.get_model()
        self.test_model(model)
        self.save_info()
    
    def predict(self,dataset,er=None,fn=None):
        self.load_info(fn)
        self.load_model(fn)
        model = self.model_base.get_model()
        result = self.predict_model(model,dataset)
        return result

    def load_model(self,fn=None):
        model = self.model_base.get_model()
        if fn is None:
            fn = self.base_dir
        model.load_weights(fn+f'Training/model_weights_'+model.name).expect_partial()
    
    def train_model(self,model,idx,idx_val,learning_rate=0.00001, epochs=40,patience=3):


        """Rewrite this loop for triplets with custom  """
        print(f'\033[93mStart Training: {self.id} {model.name}\033[0m')
        
        ## setup 
        filepath = self.base_dir+ f'Training/model_weights_{model.name}'

        train  = self.dataset.train_part(idx)
        val = self.dataset.train_part(idx_val)

        loss_dict = {n:tf.keras.losses.CategoricalCrossentropy() for n in model.output_names}

        opt = tf.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt,
              loss=loss_dict,
              metrics=['accuracy'])
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                    filepath=filepath,
                                    save_weights_only=True,
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True,
                                    verbose=1, 
                                    restore_best_weights = True)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.01, patience=patience,mode='auto')
        history = model.fit(train,validation_data=val, epochs=epochs, 
                        callbacks=[es,checkpoint])
        model.load_weights(filepath)
        return history, filepath

    def test_model(self,model,save=True,conditions=None):
        print(f'\033[93mEvaluating: {self.id} {model.name}\033[0m')
       
        stateful_metrics = ["accuracy"]
        progBar = tf.keras.utils.Progbar(len(self.dataset.test(conditions)),stateful_metrics=stateful_metrics)

        FLAG_First = True
        res={
            'Y':None, # Not usefull now, maybe later? 
            'Z':None, # Filters for nice graphs
            'R':None  # Rightfully estimated
        }

        for (x,y) in self.dataset.test(conditions):
            y_est = model(x,training=False)
            ## Calculate accuracy for each exit
            y = tf.math.argmax(y,axis=1,output_type=tf.dtypes.int32)
            y_est_val = tf.math.argmax(y_est,axis=1,output_type=tf.dtypes.int32)
            right = tf.math.equal(y, y_est_val, name=None)

            if FLAG_First:
                FLAG_First=False
                res['Y'] = y_est
                res['R'] = right
                res['Z'] = np.array(self.dataset.get_filter(x[0])).T
            else:
                res['Y'] = np.concatenate([res['Y'],y_est], axis=0)
                res['R'] = np.concatenate([res['R'],right], axis=0)
                res['Z'] = np.concatenate([res['Z'],np.array(self.dataset.get_filter(x[0])).T], axis=0)

            progBar.add(1,[('accuracy',np.sum(res['R'])/len(res['R']))])
        
        progBar.update(len(self.dataset.test(conditions)), values=[], finalize=True)

        self.info['Results'][model.name] = {'acc':np.sum(res['R'])/len(res['R'])}
        if save:
            np.save(self.base_dir+ f'Results/res_{model.name}',res)

    def predict_model(self,model,dataset):
        progBar = tf.keras.utils.Progbar(len(dataset))
        FLAG_First = True
        res = None

        for (x,y) in dataset:
            y_est = model(x,training=False)
            if FLAG_First:
                FLAG_First=False
                res = y_est
            else:
                res = np.concatenate([res,y_est], axis=0)
            progBar.add(1)
        progBar.update(len(dataset), values=[], finalize=True)
        return res,{}
