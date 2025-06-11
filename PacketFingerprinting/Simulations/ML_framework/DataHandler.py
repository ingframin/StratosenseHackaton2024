import h5py
import tensorflow as tf
import numpy as np
from itertools import product
import random
import pandas as pd
import os
from dataclasses import dataclass, field

class DataHandler_Base:
    """
       This dataset handler takes in a dataset and produces triplets.
    """   
    def __init__(self,filename,split=(75,12.5,12.5), seed=None,batchsize=128,input_shape=512) -> None:
        self.filename=filename
        self.batchsize = batchsize
        self.input_shape=input_shape
        print(f'\033[93mOpening Dataset\033[0m')
        self.data =self.extract_data(filename)
        filters = {'message_id':self.data["filters"]}
        self.idx = IDX(filters,len(self.data["filters"]),split,None,seed)
        # print(f'\033[93mGenerated {len(self.data["filters"])} triplets {(len(self.idx.train),len(self.idx.val),len(self.idx.test))} \033[0m')

        
    """ Default data extraction, makes a dictionary in which the data is structured the same as the the given structure """
    def extract_data(self,fn):
        data = {'A':None,'P':None,'N':None,'filters':None}
        return data

    def get_dataset(self,idx):
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['P'][idx,:],self.data['N'][idx,:]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)
    

    @property
    def train(self):
        return self.get_dataset(self.idx.train)

    @property
    def val(self):
        return self.get_dataset(self.idx.val)

    def test(self,conditions=None):
        return self.get_dataset(self.idx.get_test_subset_idx(conditions))
    
    def test_len(self,conditions=None):
        return len(self.idx.get_test_subset_idx(conditions))
    
    def train_part(self,idx):
        return self.get_dataset(idx)


class DataHandlerFranco(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""

    def extract_data(self, fn,umes=32,repmes=16,drep=4):
        def dataToFrame(data,lenght=512):
            data = data[:lenght]
            return np.array([np.real(data),np.imag(data)]).T
        
        d = self.h5ToDict(fn)
        data = {'A':[],'P':[],'N':[],'filters':[]}
        for m in range(umes):
            for f in range(repmes):
                for _ in range(drep):
                    p_index= random.choice([i for i in range(repmes) if i not in [f]])
                    n_set = random.choice([i for i in range(umes) if i not in [m]])
                    n_index = random.choice(range(repmes))
                    data['A'].append(dataToFrame(d[m][f]))
                    data['P'].append(dataToFrame(d[m][p_index]))
                    data['N'].append(dataToFrame(d[n_set][n_index]))
                    data['filters'].append(m)
        return {k:np.array(v) for k,v in data.items()}

    def h5ToDict(self,fn='Dataset/fsk_data.h5',unique_messages=32):
        d={}
        with h5py.File(fn, 'r') as ds:
            for i in range(unique_messages):
                d[i]=np.array(ds[str(i)])
        return d
        
class DataHandlerMatthias(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""

    def get_packet(self,i):
        def fix_length(series, size=512):
            if len(series) > size:
                return series[:size]
            elif len(series) < size:
                res = np.zeros(512,dtype=np.complex64)
                res[:len(series)]=series
                return res
            else:
                return series

        startIdx = self.detections.startIdx.loc[i]
        endIdx = self.detections.endIdx.loc[i]
        packet = np.array(self.iq[startIdx:endIdx+1].IQ)
        packet = fix_length(packet)
        power = np.mean(np.abs(packet)**2)
        return packet / np.sqrt(power)

    def extract_data(self, fn='../Datasets/Training/'):
        self.detections = pd.read_csv(fn+'1030_IQ_10s_FRA_1_detections.csv.gz',index_col=0)
        raw = np.fromfile(fn+'1030_IQ_10s_FRA_1.bin', dtype=np.int16).reshape(-1, 2)
        iq = pd.DataFrame(raw, columns=['I', 'Q'])
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq

        df = pd.read_csv(fn+'1030_IQ_10s_FRA_1_triplets.csv.gz',index_col=0)

        def dataToFrame(data,lenght=512):
            data = data[:lenght]
            return np.array([np.real(data),np.imag(data)]).T
        
        data = {'A':[],'P':[],'N':[],'filters':[],'info':[]}
        for _,(i,j,k,ci,cj,ck) in df.iterrows():
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['P'].append(dataToFrame(self.get_packet(j)))
            data['N'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            data['info'].append([ci,cj,ck])
        return {k:np.array(v) for k,v in data.items()}
    

class DataHandlerMatthiasEvaluate(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""

    def get_packet(self,i):
        def fix_length(series, size=512):
            if len(series) > size:
                return series[:size]
            elif len(series) < size:
                res = np.zeros(512,dtype=np.complex64)
                res[:len(series)]=series
                return res
            else:
                return series

        startIdx = self.detections.startIdx.loc[i]
        endIdx = self.detections.endIdx.loc[i]
        packet = np.array(self.iq[startIdx:endIdx+1].IQ)
        packet = fix_length(packet)
        power = np.mean(np.abs(packet)**2)
        return packet / np.sqrt(power)
    
    def get_dataset(self,idx):
        idx = range(len(self.data['Y']))
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['T'][idx,:],self.data['Y'][idx,:]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)

    def extract_data(self, fn='../Datasets/Validation'):
        self.detections = pd.read_csv(fn+'1030_IQ_10s_FRA_2_detections.csv.gz',index_col=0)
        raw = np.fromfile(fn+'1030_IQ_10s_FRA_2.bin', dtype=np.int16).reshape(-1, 2)
        iq = pd.DataFrame(raw, columns=['I', 'Q'])
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq

        df = pd.read_csv(fn+'1030_IQ_10s_FRA_2_triplets.csv.gz',index_col=0)

        def dataToFrame(data,lenght=512):
            data = data[:lenght]
            return np.array([np.real(data),np.imag(data)]).T
        
        data = {'A':[],'T':[],'Y':[],'filters':[],'info':[]}
        c = 0
        for _,(i,j,k,ci,cj,ck) in df.iterrows():

            # if c%2==0:
            # Positive
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['filters'].append(1)
            data['info'].append(ci)
            data['Y'].append([ci,i,j,1])
            # elif c%2==1:
            # Negative
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            data['info'].append(cj)
            data['Y'].append([cj,i,k,0])

            # Negative 2
            data['A'].append(dataToFrame(self.get_packet(j)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            data['info'].append(ck)
            data['Y'].append([ck,j,k,0])
            # c+=1


        return {k:np.array(v) for k,v in data.items()}


class DataHandlerMatthiasEvaluate2(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""

    def get_packet(self,i):
        def fix_length(series, size=512):
            if len(series) > size:
                return series[:size]
            elif len(series) < size:
                res = np.zeros(512,dtype=np.complex64)
                res[:len(series)]=series
                return res
            else:
                return series

        startIdx = self.detections.startIdx.loc[i]
        endIdx = self.detections.endIdx.loc[i]
        packet = np.array(self.iq[startIdx:endIdx+1].IQ)
        packet = fix_length(packet)
        power = np.mean(np.abs(packet)**2)
        return packet / np.sqrt(power)
    
    def get_dataset(self,idx):
        idx = range(len(self.data['Y']))
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['T'][idx,:],self.data['Y'][idx,:]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)

    def extract_data(self, fn='../Datasets/Validation'):
        self.detections = pd.read_csv(fn+'1030_IQ_10s_FRA_2_detections.csv.gz',index_col=0)
        raw = np.fromfile(fn+'1030_IQ_10s_FRA_2.bin', dtype=np.int16).reshape(-1, 2)
        iq = pd.DataFrame(raw, columns=['I', 'Q'])
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq

        df_org = pd.read_csv(fn+'1030_IQ_10s_FRA_2_correlation.csv.gz')
        samples = 1_000_000
        df = df_org.sample(n=samples,random_state=1)
        print(df.head())

        def dataToFrame(data,lenght=512):
            data = data[:lenght]
            return np.array([np.real(data),np.imag(data)]).T
        
        data = {'A':[],'T':[],'Y':[],'filters':[],'info':[]}
        progBar = tf.keras.utils.Progbar(samples)
        for _,(i,j,ci) in df.iterrows():
            progBar.add(1)

            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['info'].append(ci)
            data['filters'].append(0)
            data['Y'].append([ci,i,j,1 if ci>0.95 else 0])
            
        return {k:np.array(v) for k,v in data.items()}
    
class DataHandlerMatthiasContast(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""

    def get_packet(self,i):
        def fix_length(series, size=512):
            if len(series) > size:
                return series[:size]
            elif len(series) < size:
                res = np.zeros(512,dtype=np.complex64)
                res[:len(series)]=series
                return res
            else:
                return series

        startIdx = self.detections.startIdx.loc[i]
        endIdx = self.detections.endIdx.loc[i]
        packet = np.array(self.iq[startIdx:endIdx+1].IQ)
        packet = fix_length(packet)
        power = np.mean(np.abs(packet)**2)
        return packet / np.sqrt(power)
    
    def get_dataset(self,idx):
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['T'][idx,:],self.data['Y'][idx,:]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)

    def extract_data(self, fn='../Datasets/Training/'):
        self.detections = pd.read_csv(fn+'1030_IQ_10s_FRA_1_detections.csv.gz',index_col=0)
        raw = np.fromfile(fn+'1030_IQ_10s_FRA_1.bin', dtype=np.int16).reshape(-1, 2)
        iq = pd.DataFrame(raw, columns=['I', 'Q'])
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq

        df = pd.read_csv(fn+'1030_IQ_10s_FRA_1_triplets.csv.gz',index_col=0)

        def dataToFrame(data,lenght=512):
            data = data[:lenght]
            return np.array([np.real(data),np.imag(data)]).T
        
        data = {'A':[],'T':[],'Y':[],'filters':[],'info':[]}
        c = 0
        for _,(i,j,k,ci,cj,ck) in df.iterrows():
            # if c%2==0:
            # Positive
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['filters'].append(1)
            data['info'].append(ci)
            data['Y'].append([ci,1])
            # elif c%2==1:
            # Negative
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            data['info'].append(cj)
            data['Y'].append([cj,0])

            # Negative 2
            data['A'].append(dataToFrame(self.get_packet(j)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            data['info'].append(ck)
            data['Y'].append([ck,0])
            # c+=1

        return {k:np.array(v) for k,v in data.items()}


class DataHandlerEWSN(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""

    def __init__(self, filename, split=(75, 12.5, 12.5), seed=None, batchsize=128, input_shape=512):
        super().__init__(filename, split, seed, batchsize, input_shape)
        self.idx.load(filename+f'idx_{self.input_shape}.npz')

    def get_packet(self,i):
        def fix_length(series):
            size=self.input_shape
            if len(series) > size:
                return series[:size]
            elif len(series) < size:
                res = np.zeros(size,dtype=np.complex64)
                res[:len(series)]=series
                return res
            else:
                return series

        startIdx = self.detections.startIdx.loc[i]
        endIdx = self.detections.endIdx.loc[i]
        packet = np.array(self.iq[startIdx:endIdx+1].IQ)
        packet = fix_length(packet)
        power = np.mean(np.abs(packet)**2)
        return packet / np.sqrt(power)
    
    def get_dataset(self,idx):
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['T'][idx,:],self.data['Y'][idx,:]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)

    def extract_data(self, fn='../Dataset/EWSN/'):

        if os.path.exists(fn+f'pre_{self.input_shape}.npy'):
            # load numpy file here for data, is faster!
            data = np.load(fn+f'pre_{self.input_shape}.npy', allow_pickle=True).item()
            return data

        # self.detections = pd.read_csv(fn+'1030_IQ_10s_FRA_1_detections.csv.gz',index_col=0)
        self.detections = pd.read_pickle(fn+'Okriftel_detections.pkl.gz', compression='gzip')
        iq = pd.read_parquet(fn+'Okriftel_iq.pq.zst')
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq

        # load training data
        df = pd.read_csv(fn+'Okriftel_triplets.csv.gz',index_col=0, compression='gzip')
        def dataToFrame(data):
            return np.array([np.real(data),np.imag(data)]).T
        data = {'A':[],'T':[],'Y':[],'filters':[],'info':[]}
        for _,(i,j,k,cij,cik,cjk,hij,hik,hjk) in df.iterrows():
            # if c%2==0:
            # Positive
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['filters'].append(1)
            # data['info'].append(ci)
            data['Y'].append([cij,hij,1])
            # elif c%2==1:
            # Negative
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(cj)
            data['Y'].append([cik,hik,0])

            # Negative 2
            data['A'].append(dataToFrame(self.get_packet(j)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(ck)
            data['Y'].append([cjk,hjk,0])
            # c+=1
        
        t = len(data['A'])
        # load training data
        df = pd.read_csv(fn+'Okriftel_triplets_eval.csv.gz',index_col=0, compression='gzip')
        for _,(i,j,k,cij,cik,cjk,hij,hik,hjk) in df.iterrows():
            # if c%2==0:
            # Positive
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['filters'].append(1)
            # data['info'].append(ci)
            data['Y'].append([cij,hij,1])
            # elif c%2==1:
            # Negative
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(cj)
            data['Y'].append([cik,hik,0])

            # Negative 2
            data['A'].append(dataToFrame(self.get_packet(j)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(ck)
            data['Y'].append([cjk,hjk,0])
            # c+=1

        v = int((len(data['A'])-t)*0.2)+t
        tes = len(data['A'])

        idx_train = np.arange(t)
        print('train',idx_train[0],idx_train[-1])
        idx_val = np.arange(t,v)
        print('val',idx_val[0],idx_val[-1])
        idx_test = np.arange(v,tes)
        print('teat',idx_test[0],idx_test[-1])


        res = {k:np.array(v) for k,v in data.items()}
        np.save(fn+f'pre_{self.input_shape}', res, allow_pickle=True)
        np.savez(fn+f'idx_{self.input_shape}', train=idx_train, val=idx_val, test=idx_test)
        return {k:np.array(v) for k,v in data.items()}

class DataHandlerEWSN_Disjunct(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""

    def __init__(self, filename, split=(75, 12.5, 12.5), seed=None, batchsize=128, input_shape=512, uid='d2'):
        self.uid=uid
        super().__init__(filename, split, seed, batchsize, input_shape)
        self.idx.load(filename+f'idx_{self.uid}_{self.input_shape}.npz')

    def get_packet(self,i):
        def fix_length(series):
            size=self.input_shape
            if len(series) > size:
                return series[:size]
            elif len(series) < size:
                res = np.zeros(size,dtype=np.complex64)
                res[:len(series)]=series
                return res
            else:
                return series

        startIdx = self.detections.startIdx.loc[i] 
        if self.input_shape<129:
            startIdx+=64
        endIdx = self.detections.endIdx.loc[i]
        packet = np.array(self.iq[startIdx:endIdx+1].IQ)
        packet = fix_length(packet)
        power = np.mean(np.abs(packet)**2)
        return packet / np.sqrt(power)
    
    def get_dataset(self,idx):
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['T'][idx,:],self.data['Y'][idx,:]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)

    def extract_data(self, fn='../Dataset/EWSN/'):

        if os.path.exists(fn+f'pre_{self.uid}_{self.input_shape}.npy'):
            # load numpy file here for data, is faster!
            data = np.load(fn+f'pre_{self.uid}_{self.input_shape}.npy', allow_pickle=True).item()
            return data

        # self.detections = pd.read_csv(fn+'1030_IQ_10s_FRA_1_detections.csv.gz',index_col=0)
        self.detections = pd.read_pickle(fn+'Okriftel_detections.pkl.gz', compression='gzip')
        iq = pd.read_parquet(fn+'Okriftel_iq.pq.zst')
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq

        # load training data
        df = pd.read_csv(fn+'dieter_triplets_train.csv.gz',index_col=0, compression='gzip')
        def dataToFrame(data):
            return np.array([np.real(data),np.imag(data)]).T
        data = {'A':[],'T':[],'Y':[],'filters':[],'info':[]}
        for _,(i,j,k,cij,cik,cjk,hij,hik,hjk) in df.iterrows():
            # if c%2==0:
            # Positive
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['filters'].append(1)
            # data['info'].append(ci)
            data['Y'].append([cij,hij,1])
            # elif c%2==1:
            # Negative
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(cj)
            data['Y'].append([cik,hik,0])

            # Negative 2
            data['A'].append(dataToFrame(self.get_packet(j)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(ck)
            data['Y'].append([cjk,hjk,0])
            # c+=1
        
        t = len(data['A'])
        
        # load val data
        df = pd.read_csv(fn+'dieter_triplets_val.csv.gz',index_col=0, compression='gzip')
        for _,(i,j,k,cij,cik,cjk,hij,hik,hjk) in df.iterrows():
            # if c%2==0:
            # Positive
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['filters'].append(1)
            # data['info'].append(ci)
            data['Y'].append([cij,hij,1])
            # elif c%2==1:
            # Negative
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(cj)
            data['Y'].append([cik,hik,0])

            # Negative 2
            data['A'].append(dataToFrame(self.get_packet(j)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(ck)
            data['Y'].append([cjk,hjk,0])
            # c+=1

        v = len(data['A'])

         # load val data
        df = pd.read_csv(fn+'dieter_triplets_eval.csv.gz',index_col=0, compression='gzip')
        for _,(i,j,k,cij,cik,cjk,hij,hik,hjk) in df.iterrows():
            # if c%2==0:
            # Positive
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['filters'].append(1)
            # data['info'].append(ci)
            data['Y'].append([cij,hij,1])
            # elif c%2==1:
            # Negative
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(cj)
            data['Y'].append([cik,hik,0])

            # Negative 2
            data['A'].append(dataToFrame(self.get_packet(j)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(ck)
            data['Y'].append([cjk,hjk,0])
            # c+=1

        tes = len(data['A'])

        idx_train = np.arange(t)
        print('train',idx_train[0],idx_train[-1])
        idx_val = np.arange(t,v)
        print('val',idx_val[0],idx_val[-1])
        idx_test = np.arange(v,tes)
        print('teat',idx_test[0],idx_test[-1])


        res = {k:np.array(v) for k,v in data.items()}
        np.save(fn+f'pre_{self.uid}_{self.input_shape}', res, allow_pickle=True)
        np.savez(fn+f'idx_{self.uid}_{self.input_shape}', train=idx_train, val=idx_val, test=idx_test)
        return {k:np.array(v) for k,v in data.items()}

class DataHandlerEWSN_Disjunct2(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""

    def __init__(self, filename, split=(75, 12.5, 12.5), seed=None, batchsize=128, input_shape=512, uid='d2'):
        self.uid=uid
        super().__init__(filename, split, seed, batchsize, input_shape)
        self.idx.load(filename+f'idx_{self.uid}_{self.input_shape}.npz')

    def get_packet(self,i):
        def fix_length(series):
            size=self.input_shape
            if len(series) > size:
                return series[:size]
            elif len(series) < size:
                res = np.zeros(size,dtype=np.complex64)
                res[:len(series)]=series
                return res
            else:
                return series

        startIdx = self.detections.startIdx.loc[i] 
        # if self.input_shape<129:
        #     startIdx+=64
        ds = 256//self.input_shape ## downsample test
        endIdx = self.detections.endIdx.loc[i]
        packet = np.array(self.iq[startIdx:endIdx+1:ds].IQ)
        packet = fix_length(packet)
        power = np.mean(np.abs(packet)**2)
        return packet / np.sqrt(power)
    
    def get_dataset(self,idx):
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['T'][idx,:],self.data['Y'][idx,:]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)

    def extract_data(self, fn='../Dataset/EWSN_disjunct/'):

        if os.path.exists(fn+f'pre_{self.uid}_{self.input_shape}.npy'):
            # load numpy file here for data, is faster!
            data = np.load(fn+f'pre_{self.uid}_{self.input_shape}.npy', allow_pickle=True).item()
            return data

        # self.detections = pd.read_csv(fn+'1030_IQ_10s_FRA_1_detections.csv.gz',index_col=0)
        # self.detections = pd.read_pickle(fn+'Okriftel_detections.pkl.gz', compression='gzip')
        # iq = pd.read_parquet(fn+'Okriftel_iq.pq.zst')
        # iq['IQ'] = iq.I + 1j*iq.Q
        # self.iq=iq

        # load training data
        mode = 'Training'
        df = pd.read_csv(fn+f'Okriftel_{mode}_triplets.csv.gz',index_col=0, compression='gzip')
        self.detections = pd.read_pickle(fn+f'Okriftel_{mode}_detections.pkl.gz', compression='gzip')
        iq = pd.read_parquet(fn+f'Okriftel_{mode}_iq.pq.zst')
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq

        def dataToFrame(data):
            return np.array([np.real(data),np.imag(data)]).T
        data = {'A':[],'T':[],'Y':[],'filters':[],'info':[]}
        for _,(i,j,k,cij,cik,cjk,hij,hik,hjk) in df.iterrows():
            # if c%2==0:
            # Positive
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['filters'].append(1)
            # data['info'].append(ci)
            data['Y'].append([cij,hij,1])
            # elif c%2==1:
            # Negative
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(cj)
            data['Y'].append([cik,hik,0])

            # Negative 2
            data['A'].append(dataToFrame(self.get_packet(j)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(ck)
            data['Y'].append([cjk,hjk,0])
            # c+=1
        
        t = len(data['A'])
        
        # load val data
        mode = 'Validation'
        df = pd.read_csv(fn+f'Okriftel_{mode}_triplets.csv.gz',index_col=0, compression='gzip')
        self.detections = pd.read_pickle(fn+f'Okriftel_{mode}_detections.pkl.gz', compression='gzip')
        iq = pd.read_parquet(fn+f'Okriftel_{mode}_iq.pq.zst')
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq
        # df = pd.read_csv(fn+'dieter_triplets_val.csv.gz',index_col=0, compression='gzip')
        for _,(i,j,k,cij,cik,cjk,hij,hik,hjk) in df.iterrows():
            # if c%2==0:
            # Positive
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['filters'].append(1)
            # data['info'].append(ci)
            data['Y'].append([cij,hij,1])
            # elif c%2==1:
            # Negative
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(cj)
            data['Y'].append([cik,hik,0])

            # Negative 2
            data['A'].append(dataToFrame(self.get_packet(j)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(ck)
            data['Y'].append([cjk,hjk,0])
            # c+=1

        v = len(data['A'])

         # load val data
        mode = 'Testing'
        df = pd.read_csv(fn+f'Okriftel_{mode}_triplets.csv.gz',index_col=0, compression='gzip')
        self.detections = pd.read_pickle(fn+f'Okriftel_{mode}_detections.pkl.gz', compression='gzip')
        iq = pd.read_parquet(fn+f'Okriftel_{mode}_iq.pq.zst')
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq
        for _,(i,j,k,cij,cik,cjk,hij,hik,hjk) in df.iterrows():
            # if c%2==0:
            # Positive
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['filters'].append(1)
            # data['info'].append(ci)
            data['Y'].append([cij,hij,1])
            # elif c%2==1:
            # Negative
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(cj)
            data['Y'].append([cik,hik,0])

            # Negative 2
            data['A'].append(dataToFrame(self.get_packet(j)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            # data['info'].append(ck)
            data['Y'].append([cjk,hjk,0])
            # c+=1

        tes = len(data['A'])

        idx_train = np.arange(t)
        print('train',idx_train[0],idx_train[-1])
        idx_val = np.arange(t,v)
        print('val',idx_val[0],idx_val[-1])
        idx_test = np.arange(v,tes)
        print('teat',idx_test[0],idx_test[-1])


        res = {k:np.array(v) for k,v in data.items()}
        np.save(fn+f'pre_{self.uid}_{self.input_shape}', res, allow_pickle=True)
        np.savez(fn+f'idx_{self.uid}_{self.input_shape}', train=idx_train, val=idx_val, test=idx_test)
        return {k:np.array(v) for k,v in data.items()}

class DataHandlerEWSN_Evaluation(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""

    def __init__(self, filename, split=(75, 12.5, 12.5), seed=None, batchsize=128, input_shape=512, uid='eval'):
        self.uid=uid
        super().__init__(filename, split, seed, batchsize, input_shape)
        self.idx.load(filename+f'idx_{self.uid}_{self.input_shape}.npz')

    def get_packet(self,i):
        def fix_length(series):
            size=self.input_shape
            if len(series) > size:
                return series[:size]
            elif len(series) < size:
                res = np.zeros(size,dtype=np.complex64)
                res[:len(series)]=series
                return res
            else:
                return series

        startIdx = self.detections.startIdx.loc[i] 
        # if self.input_shape<129:
        #     startIdx+=64
        ds = 256//self.input_shape ## downsample test
        endIdx = self.detections.endIdx.loc[i]
        packet = np.array(self.iq[startIdx:endIdx+1:ds].IQ)
        packet = fix_length(packet)
        power = np.mean(np.abs(packet)**2)
        return packet / np.sqrt(power)
    
    def get_dataset(self,idx):
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['A'][idx,:],self.data['Y'][idx,:]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)

    def extract_data(self, fn='../Dataset/EWSN_disjunct/'):

        if os.path.exists(fn+f'pre_{self.uid}_{self.input_shape}.npy'):
            # load numpy file here for data, is faster!
            data = np.load(fn+f'pre_{self.uid}_{self.input_shape}.npy', allow_pickle=True).item()
            return data


        # load training data
        def dataToFrame(data):
            return np.array([np.real(data),np.imag(data)]).T
        data = {'A':[],'T':[],'Y':[],'filters':[],'info':[]}
        

         # load val data
        mode = 'Testing' if self.uid == 'eval' else 'Validation'
        print(mode)
        self.detections = pd.read_pickle(fn+f'Okriftel_{mode}_detections.pkl.gz', compression='gzip')
        iq = pd.read_parquet(fn+f'Okriftel_{mode}_iq.pq.zst')
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq

        # add all detections. 
        df = pd.read_csv(fn+f'Okriftel_{mode}_correlation.csv.gz', compression='gzip')
        dd = set(pd.concat([df['i'], df['j']]))


        for i in dd:
            # if c%2==0:
            # Positive
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['filters'].append(1)
            # data['info'].append(ci)
            data['Y'].append([i,i,1])
            # elif c%2==1:

        tes = len(data['A'])

        idx_train = np.arange(3)
        print('train',idx_train[0],idx_train[-1])
        idx_val = np.arange(3,7)
        print('val',idx_val[0],idx_val[-1])
        idx_test = np.arange(tes)
        print('teat',idx_test[0],idx_test[-1])


        res = {k:np.array(v) for k,v in data.items()}
        np.save(fn+f'pre_{self.uid}_{self.input_shape}', res, allow_pickle=True)
        np.savez(fn+f'idx_{self.uid}_{self.input_shape}', train=idx_train, val=idx_val, test=idx_test)
        return {k:np.array(v) for k,v in data.items()}

class DataHandlerEWSN2(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""

    def __init__(self, filename, split=(75, 12.5, 12.5), seed=None, batchsize=128, input_shape=512):
        super().__init__(filename, split, seed, batchsize, input_shape)
        self.idx.load(filename+f'idx_2_{self.input_shape}.npz')
        print(f'\033[93mGenerated {len(self.data["filters"])} triplets {(len(self.idx.train),len(self.idx.val),len(self.idx.test))} \033[0m')


    def get_packet(self,startIdx):
        packet = np.array(self.iq[startIdx:startIdx+self.input_shape].IQ)
        power = np.mean(np.abs(packet)**2)
        return packet / np.sqrt(power)
    
    def get_dataset(self,idx):
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['T'][idx,:],self.data['Y'][idx,:]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)

    def extract_data(self, fn='../Dataset/EWSN/'):

        if os.path.exists(fn+f'pre_2_{self.input_shape}.npy'):
            # load numpy file here for data, is faster!
            data = np.load(fn+f'pre_2_{self.input_shape}.npy', allow_pickle=True).item()
            return data

        # self.detections = pd.read_csv(fn+'1030_IQ_10s_FRA_1_detections.csv.gz',index_col=0)
        # self.detections = pd.read_pickle(fn+'Okriftel_detections.pkl.gz', compression='gzip')
        iq = pd.read_parquet(fn+'Okriftel_iq.pq.zst')
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq

        def dataToFrame(data):
            return np.array([np.real(data),np.imag(data)]).T
        data = {'A':[],'T':[],'Y':[],'filters':[],'info':[]}

        # load training data
        idx = np.load(fn+'train.npz')
        # positives
        for idx_a, idx_p in idx['pos']:
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p)))
            data['filters'].append(1)
            data['Y'].append([0,0,1])
        # negatives
        for idx_a, idx_p in idx['neg']:
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p)))
            data['filters'].append(0)
            data['Y'].append([0,0,0])
        t = len(idx['pos'])+len(idx['neg'])
        
        # load Val data
        idx = np.load(fn+'val.npz')
        # positives
        for idx_a, idx_p in idx['pos']:
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p)))
            data['filters'].append(1)
            data['Y'].append([0,0,1])
        # negatives
        for idx_a, idx_p in idx['neg']:
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p)))
            data['filters'].append(0)
            data['Y'].append([0,0,0])
        v = len(idx['pos'])+len(idx['neg'])

        # load tst data
        idx = np.load(fn+'test.npz')
        # positives
        for idx_a, idx_p in idx['pos']:
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p)))
            data['filters'].append(1)
            data['Y'].append([0,0,1])
        # negatives
        for idx_a, idx_p in idx['neg']:
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p)))
            data['filters'].append(0)
            data['Y'].append([0,0,0])
        tst = len(idx['pos'])+len(idx['neg'])


        idx_train = np.arange(t)
        print('train',idx_train[0],idx_train[-1])
        idx_val = np.arange(t,v+t)
        print('val',idx_val[0],idx_val[-1])
        idx_test = np.arange(v+t,tst+v+t)
        print('teat',idx_test[0],idx_test[-1])


        res = {k:np.array(v) for k,v in data.items()}
        np.save(fn+f'pre_2_{self.input_shape}', res, allow_pickle=True)
        np.savez(fn+f'idx_2_{self.input_shape}', train=idx_train, val=idx_val, test=idx_test)
        return {k:np.array(v) for k,v in data.items()}


class DataHandlerEWSN_shift(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""
    # make sure packets are not as well aligned
    def __init__(self, filename, split=(75, 12.5, 12.5), seed=None, batchsize=128, input_shape=512,uid='shift'):
        self.uid=uid
        super().__init__(filename, split, seed, batchsize, input_shape)
        self.idx.load(filename+f'idx_{self.uid}_{self.input_shape}.npz')
        print(f'\033[93mGenerated {len(self.data["filters"])} triplets {(len(self.idx.train),len(self.idx.val),len(self.idx.test))} \033[0m')


    def get_packet(self,startIdx):
        packet = np.array(self.iq[startIdx:startIdx+self.input_shape].IQ)
        power = np.mean(np.abs(packet)**2)
        return packet / np.sqrt(power)
    
    def get_dataset(self,idx):
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['T'][idx,:],self.data['Y'][idx,:]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)

    def extract_data(self, fn='../Dataset/EWSN/'):

        if os.path.exists(fn+f'pre_{self.uid}_{self.input_shape}.npy'):
            # load numpy file here for data, is faster!
            data = np.load(fn+f'pre_{self.uid}_{self.input_shape}.npy', allow_pickle=True).item()
            return data

        # self.detections = pd.read_csv(fn+'1030_IQ_10s_FRA_1_detections.csv.gz',index_col=0)
        # self.detections = pd.read_pickle(fn+'Okriftel_detections.pkl.gz', compression='gzip')
        iq = pd.read_parquet(fn+'Okriftel_iq.pq.zst')
        iq['IQ'] = iq.I + 1j*iq.Q
        self.iq=iq

        def dataToFrame(data):
            return np.array([np.real(data),np.imag(data)]).T
        data = {'A':[],'T':[],'Y':[],'filters':[],'info':[]}

        # load training data
        idx = np.load(fn+'train.npz')
        # positives
        for idx_a, idx_p in idx['pos']:
            l = np.random.randint(-2,2)
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p+l)))
            data['filters'].append(1)
            data['Y'].append([0,0,1])
        # negatives
        for idx_a, idx_p in idx['neg']:
            l = np.random.randint(-2,2)
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p+l)))
            data['filters'].append(0)
            data['Y'].append([0,0,0])
        t = len(idx['pos'])+len(idx['neg'])
        
        # load Val data
        idx = np.load(fn+'val.npz')
        # positives
        for idx_a, idx_p in idx['pos']:
            l = np.random.randint(-2,2)
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p+l)))
            data['filters'].append(1)
            data['Y'].append([0,0,1])
        # negatives
        for idx_a, idx_p in idx['neg']:
            l = np.random.randint(-2,2)
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p+l)))
            data['filters'].append(0)
            data['Y'].append([0,0,0])
        v = len(idx['pos'])+len(idx['neg'])

        # load tst data
        idx = np.load(fn+'test.npz')
        # positives
        for idx_a, idx_p in idx['pos']:
            l = np.random.randint(-2,2)
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p+l)))
            data['filters'].append(1)
            data['Y'].append([0,0,1])
        # negatives
        for idx_a, idx_p in idx['neg']:
            l = np.random.randint(-2,2)
            data['A'].append(dataToFrame(self.get_packet(idx_a)))
            data['T'].append(dataToFrame(self.get_packet(idx_p+l)))
            data['filters'].append(0)
            data['Y'].append([0,0,0])
        tst = len(idx['pos'])+len(idx['neg'])


        idx_train = np.arange(t)
        print('train',idx_train[0],idx_train[-1])
        idx_val = np.arange(t,v+t)
        print('val',idx_val[0],idx_val[-1])
        idx_test = np.arange(v+t,tst+v+t)
        print('teat',idx_test[0],idx_test[-1])


        res = {k:np.array(v) for k,v in data.items()}
        np.save(fn+f'pre_{self.uid}_{self.input_shape}', res, allow_pickle=True)
        np.savez(fn+f'idx_{self.uid}_{self.input_shape}', train=idx_train, val=idx_val, test=idx_test)
        return {k:np.array(v) for k,v in data.items()}




import random 

class DataHandlerAMC:
    """
        DataHandler handles data of a HDF5 dataset. This serves as a base, overwrite if needed.
        It loads all data and parameters of the dataset as well as distributing the data into train, validate and test.
    """   
    def __init__(self,filename,split=(50,25,25), seed=None,batchsize=64,conditions=None, input_shape=256, id='n') -> None:
        self.filename=filename
        self.input_shape= input_shape
        self.batchsize = batchsize
        self.uid=id
        self.params, self.structure = self.get_params(filename+'Generated_modlations.h5')

        if os.path.exists(self.filename+f'pre_{self.uid}_{self.input_shape}.npy'):
            # load numpy file here for data, is faster!
            self.data = np.load(self.filename+f'pre_{self.uid}_{self.input_shape}.npy', allow_pickle=True).item()
        else:
            self.data =self.extract_data(filename+'Generated_modlations.h5')
            filters = {f:self.data["Filters"][f] for f in ['modulation','SNR']}
            self.idx = IDX(filters,self.params.all_frames,split,conditions,seed)
            self.data = self.post_build_ds()
        filters = {'message_id':self.data["filters"]}
        self.idx = IDX(filters,self.params.all_frames,split,conditions,seed)
        self.idx.load(filename+f'idx_{self.uid}_{self.input_shape}.npz')
        ## From data_temp 
        # 1. from the idx, just get some random idxs and make some random combos 
        # 2. 1 positive, 2 negative wiht same modulation, 3 random negatives

    def get_dataset(self,idx):
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['T'][idx,:],self.data['Y'][idx,:]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)
    
    def post_build_ds(self):

        data = {'A':[],'T':[],'Y':[],'filters':[],'info':[]}
        # train 
        # all idx
        all_idx = np.array(self.idx.select_idx({'modulation':range(6),'SNR':[10]}, self.idx.train))
        s = all_idx.shape[1]//2
        for i,idx in enumerate(all_idx):
            # positives
            for ii in random.sample(list(idx),s):
                data['A'].append(self.data['X_RUS']['RU_0'][ii,:self.input_shape]) ## might need transpose here 
                l = np.random.randint(0,3) if self.uid=='shift' else 0
                data['T'].append(self.data['X_RUS']['RU_1'][ii,l:self.input_shape+l]) ## might need transpose here 
                data['filters'].append(1)
                data['Y'].append([i,i,1])
            
            # Negatives own group: 
            for ii in random.sample(list(idx),s):
                idx_rem = set(idx)
                idx_rem.remove(ii)
                jj = random.choice(list(idx_rem))
                data['A'].append(self.data['X_RUS']['RU_0'][ii,:self.input_shape]) ## might need transpose here 
                data['T'].append(self.data['X_RUS']['RU_1'][jj,:self.input_shape]) ## might need transpose here 
                data['filters'].append(0)
                data['Y'].append([i,i,0])

            # Negatives not in group 
            i_idx = set(range(6))
            i_idx.remove(i)
            for j in i_idx:
                for ii in random.sample(list(idx),s//2):
                    jj = random.choice(list(all_idx[j]))
                    data['A'].append(self.data['X_RUS']['RU_0'][ii,:self.input_shape]) ## might need transpose here 
                    data['T'].append(self.data['X_RUS']['RU_1'][jj,:self.input_shape]) ## might need transpose here 
                    data['filters'].append(0)
                    data['Y'].append([i,j,0])
        t = len(data['A'])
        
        #validation
        all_idx = np.array(self.idx.select_idx({'modulation':range(6),'SNR':[10]}, self.idx.val))
        s = all_idx.shape[1]//4
        for i,idx in enumerate(all_idx):
            # positives
            for ii in random.sample(list(idx),s):
                data['A'].append(self.data['X_RUS']['RU_0'][ii,:self.input_shape]) ## might need transpose here 
                l = np.random.randint(0,3) if self.uid=='shift' else 0
                data['T'].append(self.data['X_RUS']['RU_1'][ii,l:self.input_shape+l]) ## might need transpose here 
                data['filters'].append(1)
                data['Y'].append([i,i,1])
            
            # Negatives own group: 
            for ii in random.sample(list(idx),s):
                idx_rem = set(idx)
                idx_rem.remove(ii)
                jj = random.choice(list(idx_rem))
                data['A'].append(self.data['X_RUS']['RU_0'][ii,:self.input_shape]) ## might need transpose here 
                data['T'].append(self.data['X_RUS']['RU_1'][jj,:self.input_shape]) ## might need transpose here 
                data['filters'].append(0)
                data['Y'].append([i,j,0])

            # Negatives not in group 
            i_idx = set(range(6))
            i_idx.remove(i)
            for j in i_idx:
                for ii in random.sample(list(idx),s//2):
                    jj = random.choice(list(all_idx[j]))
                    data['A'].append(self.data['X_RUS']['RU_0'][ii,:self.input_shape]) ## might need transpose here 
                    data['T'].append(self.data['X_RUS']['RU_1'][jj,:self.input_shape]) ## might need transpose here 
                    data['filters'].append(0)
                    data['Y'].append([i,j,0])
        v = len(data['A'])-t

        #test
        all_idx = np.array(self.idx.select_idx({'modulation':range(6),'SNR':[10]}, self.idx.test))
        s = all_idx.shape[1]//4
        for i,idx in enumerate(all_idx):
            # positives
            for ii in random.sample(list(idx),s):
                data['A'].append(self.data['X_RUS']['RU_0'][ii,:self.input_shape]) ## might need transpose here 
                l = np.random.randint(0,3) if self.uid=='shift' else 0
                data['T'].append(self.data['X_RUS']['RU_1'][ii,l:self.input_shape+l]) ## might need transpose here 
                data['filters'].append(1)
                data['Y'].append([i,i,1])
            
            # Negatives own group: 
            for ii in random.sample(list(idx),s):
                idx_rem = set(idx)
                idx_rem.remove(ii)
                jj = random.choice(list(idx_rem))
                data['A'].append(self.data['X_RUS']['RU_0'][ii,:self.input_shape]) ## might need transpose here 
                data['T'].append(self.data['X_RUS']['RU_1'][jj,:self.input_shape]) ## might need transpose here 
                data['filters'].append(0)
                data['Y'].append([i,i,0])

            # Negatives not in group 
            i_idx = set(range(6))
            i_idx.remove(i)
            for j in i_idx:
                for ii in random.sample(list(idx),s//2):
                    jj = random.choice(list(all_idx[j]))
                    data['A'].append(self.data['X_RUS']['RU_0'][ii,:self.input_shape]) ## might need transpose here 
                    data['T'].append(self.data['X_RUS']['RU_1'][jj,:self.input_shape]) ## might need transpose here 
                    data['filters'].append(0)
                    data['Y'].append([i,j,0])
        tes = len(data['A'])-t-v


        idx_train = np.arange(t)
        print('train',idx_train[0],idx_train[-1])
        idx_val = np.arange(t,v+t)
        print('val',idx_val[0],idx_val[-1])
        idx_test = np.arange(v+t,tes+v+t)
        print('teat',idx_test[0],idx_test[-1])


        res = {k:np.array(v) for k,v in data.items()}
        np.save(self.filename+f'pre_{self.uid}_{self.input_shape}', res, allow_pickle=True)
        np.savez(self.filename+f'idx_{self.uid}_{self.input_shape}', train=idx_train, val=idx_val, test=idx_test)
        return {k:np.array(v) for k,v in data.items()}

    def get_params(self, fn):
        with h5py.File(fn, 'r') as ds:
            print(f'\033[93mOpening Dataset\033[0m')
            print('Description:', ds.attrs['Description'])
            att = Data_set_params(*eval(ds.attrs['Parameters']).values()) ## I know this is unsave, but the other options do not work :) Deal with it!
            struc = eval(ds.attrs['Structure'])
        return att, struc
    
    """ Default data extraction, makes a dictionary in which the data is structured the same as the the given structure """
    def extract_data(self,fn):

        ## If I made a preloaded dataset
        print(f'\033[93mLoading all data\033[0m')
        with h5py.File(fn, 'r') as ds:
            data = self.extract_data_rec(ds,self.structure)
        print(f'\033[93mData loaded successfully\033[0m')
        return data
             
    def extract_data_rec(self, ds, struct):
        d = {}
        for level in struct:
            if type(level) is tuple:
                d[level[0]] = self.extract_data_rec(ds[level[0]],level[1])
            else:
                if 'Filters' in ds[level].attrs.keys():
                    filts = eval(ds[level].attrs["Filters"])
                    data = np.array(ds[level]).T
                    d["Filters"] = {}
                    for f,dd in zip(filts,data):
                        d["Filters"][f]=dd
                else:
                    d[level] = np.array(ds[level])
        return d
    
    def get_filter(self,idx,name=False):
        vals = []
        names = []
        for k in self.data["Filters"]:
            names.append(k)
            vals.append(self.data["Filters"][k][idx])
        if name:
            return vals, names
        return vals

    @property
    def train(self):
        return self.get_dataset(self.idx.train)

    @property
    def val(self):
        return self.get_dataset(self.idx.val)

    def test(self,conditions=None):
        return self.get_dataset(self.idx.get_test_subset_idx(conditions))
    
    def test_len(self,conditions=None):
        return len(self.idx.get_test_subset_idx(conditions))
    
    def train_part(self,idx):
        return self.get_dataset(idx)


class IDX:
    """
        This class handles all data selecting and shuffeling.
        When training in multiple stages, it is possible to save and load the split between train, val and test indices, making sure there is no training testing contamination.
    """
    def __init__(self, filt_data,total_frames,split, conditions,seed):
        
        self.filt_data = filt_data
        self.split_data(conditions,split,total_frames,seed)
    
    def check_conditions(self,conditions):    
        ### If no condition is passed take all data
        if conditions is None:
            conditions = {}
            for filt in self.filt_data:
                conditions[filt] = np.unique(self.filt_data[filt])
        else:
            ## check if the keys of the conditions exitst
            for key in conditions:
                if key not in self.filt_data.keys():
                    raise Exception(f' Condition {key} is not in {self.filt_data.keys()}')  
        return conditions  
    
    def select_idx(self,conditions,idx_org):
        conditions=self.check_conditions(conditions)
        idx_res = []
        for x in product(*conditions.values()): ## Iterate over all given condition value combinations
            idx = idx_org # start with all indices and filter out the unwanted indices
            for filt, value in zip(conditions.keys(),x):
                idx_tmp = np.where(self.filt_data[filt]==value)[0]
                idx = np.intersect1d(idx, idx_tmp)
            idx_res.append(idx)
        return idx_res

    def split_data(self,conditions,split,total_frames,seed):
        split = np.array(split)/sum(split)
        train, val , test = [] ,[],[]
        np.random.seed(seed)
        idx = self.select_idx(conditions,np.arange(total_frames))
        for idx_tmp in idx:
            distr = np.cumsum(split*len(idx_tmp)).astype(int)
            train.append(idx_tmp[:distr[0]])
            val.append(idx_tmp[distr[0]:distr[1]])
            test.append(idx_tmp[distr[1]:])
        self.train = np.concatenate(train)
        self.val = np.concatenate(val)
        self.test = np.concatenate(test)
            

    def get_test_subset_idx(self,conditions=None):
        return np.concatenate(self.select_idx(conditions,self.test))

    def save(self,fn):
        np.savez(fn, train=self.train, val=self.val, test=self.test)

    def load(self,fn):
        data = np.load(fn)
        self.train=data['train']
        self.val =data['val']
        self.test =data['test']



@dataclass
class Data_set_params:
    """
        This dataclass stores all the parameters. You can either adapt them here, or change them with the initialization. 
        This is a clean way to make sure everyting stays readable.
        On top of this, you can just compile all used paramters to a dictionary, making sure you remember the parameters when needed.
        This dataset will also save which blocks have been enabled and disabled.
    """

    ## General parameters
    f_c :float =1e9 
    samp_rate:float= 1e6
    sps : int = 2

    ## Pulse shaping
    excess_bw:float=0.35
    nfilts: int = 32

    @property
    def ntaps_ps(self) ->int: return self.nfilts * 11 * self.sps    # make nfilts filters of ntaps each
    
    ## Clock 
    ppm_max:float=0.5
    ppm_std:float=1e-5
    
    @property
    def cfo_max_hz(self)-> float: return self.f_c*self.ppm_max*1e-6   
    @property
    def cfo_std_hz(self)-> float: return self.f_c*self.ppm_std*1e-6
    @property
    def sro_max_hz(self)-> float: return self.samp_rate*self.ppm_max*1e-6
    @property
    def sro_std_hz(self)-> float: return self.samp_rate*self.ppm_std*1e-6

    ## Fading_block
    delays_list: list[int] =  field(default_factory=lambda: [0, 50, 120, 200, 230, 500, 1600, 2300, 5000])
    mags_list: list[int] =  field(default_factory=lambda: [-1, -1, -1, 0, 0, 0, -3, -5, -7])
    fD: int = 70 # ETU70, max doppler ferquency/frequency devaition - 70Hz.
    ntaps: int = 8
    numSinusoids: int= 8
    Kfactor: int= 4
    
    @property
    def mags(self)->list: return [10 ** (mags_val / 20.0) for mags_val in self.mags_list]
    @property
    def delays(self)->list: return [val_temp * (1e-9) * self.samp_rate for val_temp in self.delays_list]

    ## Dataset parameters
    snr_levels: range = range(-20, 31, 2)
    modulations: list[str] = field(default_factory=lambda: ["BPSK","QPSK","PSK8","PAM4","QAM16","QAM64","WBFM","GFSK","CPFSK","AM-DSB"])
    Number_of_frames: int = 1024
    samples_frame: int = 1024
    transients:int = 512
    seed:int=168000

    @property
    def gen_nr_samps(self)-> int: return 2*self.transients+self.samples_frame

    ## Blocks
    AWGN:bool=False
    SRO:bool=False
    CFO :bool=False
    FADING :bool=False
    LOS:bool=False  # Rayleigh channel, no LOS component , rician 
    PHASE_OFFSET:bool=False

    ## Modulation parameters
    CPFSK_mod_index:float=0.5

    GFSK_BT:float =0.3 # source for BT value chosen: https://comblock.com/download/com1028.pdf
    GFSK_sensitivity:float = 1.57 # approx to pi/2.

    audio_rate:float = 44.1e3 

    WBFM_freq_dev:float=  75e3 # wideband FM freq deviation typical value - chosen from https://en.wikipedia.org/wiki/Frequency_modulation
    WBFM_tau:float = 75e-6 # preemphasis time constant (default 75e-6), value used frorm https://github.com/gnuradio/gnuradio/blob/master/gr-analog/python/analog/wfm_tx.py
    
    @property
    def analog_resample_rate(self): return self.audio_rate/self.samp_rate

    ## Usefull functions
    # gives all combinations to generate each frame
    @property
    def all_frames(self): return self.Number_of_frames*len(self.snr_levels)*len(self.modulations)

    @property
    def output_shape(self): return len(self.modulations)
    
    @property
    def dict(self): 
        d = self.__dict__
        d.pop("analog_source")
        d.pop("analog_len")
        return self.__dict__

    analog_file:str="sources/cont_source.npy"



def tree(filename):
    with h5py.File(filename, 'r') as ds:
        print(filename)
        h5_tree(ds)


def h5_tree(val, pre=''):
    items = len(val)
    for attr in val.attrs:
        print(pre+'│ ',attr+':',val.attrs[attr] )              
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + f"  {val.shape}")
                for attr in val.attrs:
                    print(pre+'  ',attr+':',val.attrs[attr] )
                
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:

                print(pre + '├── ' + key +  f"  {val.shape}") 
                for attr in val.attrs:
                    print(pre+'│ ',attr+':',val.attrs[attr] )