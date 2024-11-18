import h5py
import tensorflow as tf
import numpy as np
from itertools import product
import random
import pandas as pd

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
        print(f'\033[93mGenerated {len(self.data["filters"])} triplets {(len(self.idx.train),len(self.idx.val),len(self.idx.test))} \033[0m')

        
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
        df = df_org.sample(n=100000,random_state=1)
        print(df.head())

        def dataToFrame(data,lenght=512):
            data = data[:lenght]
            return np.array([np.real(data),np.imag(data)]).T
        
        data = {'A':[],'T':[],'Y':[],'filters':[],'info':[]}
        c = 0
        for _,(i,j,ci) in df.iterrows():

            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(j)))
            data['info'].append(ci)
            data['filters'].append(0)
            data['Y'].append([ci,i,j,1])
            
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
        ds  = tf.data.Dataset.from_tensor_slices((self.data['A'][idx,:],self.data['T'][idx,:],self.data['Y'][idx]))
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
            data['Y'].append(ci)
            # elif c%2==1:
            # Negative
            data['A'].append(dataToFrame(self.get_packet(i)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            data['info'].append(cj)
            data['Y'].append(cj)

            # Negative 2
            data['A'].append(dataToFrame(self.get_packet(j)))
            data['T'].append(dataToFrame(self.get_packet(k)))
            data['filters'].append(0)
            data['info'].append(ck)
            data['Y'].append(ck)
            # c+=1

            # # Negative 2
            # data['A'].append(dataToFrame(self.get_packet(j)))
            # data['T'].append(dataToFrame(self.get_packet(k)))
            # data['filters'].append(0)
            # data['info'].append(ck)
            # data['Y'].append(ck)

        return {k:np.array(v) for k,v in data.items()}

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