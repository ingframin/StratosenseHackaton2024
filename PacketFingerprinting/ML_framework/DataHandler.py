import h5py
import tensorflow as tf
import numpy as np
from itertools import product
from dataclasses import dataclass, field

"""Adapt this dataHandler to the data. The output of the training data should be (anchor, similar, different), for the validation and test data it should be (some signal, some different or the same singal.) """
class DataHandler:
    """
        DataHandler handles data of a dataset. This serves as a base, overwrite if needed.
        It loads all data and parameters of the dataset as well as distributing the data into train, validate and test.
    """   
    def __init__(self,filename,split=(75,12.5,12.5), seed=None,batchsize=256,conditions=None) -> None:
        self.filename=filename
        self.batchsize = batchsize
        self.params, self.structure = self.get_params(filename)
        self.data =self.extract_data(filename)
        filters = {f:self.data["Filters"][f] for f in ['modulation','SNR']}
        self.idx = IDX(filters,self.params.all_frames,split,conditions,seed)
    
    def get_params(self, fn):
        with h5py.File(fn, 'r') as ds:
            print(f'\033[93mOpening Dataset\033[0m')
            print('Description:', ds.attrs['Description'])
            att = Data_set_params(*eval(ds.attrs['Parameters']).values()) ## I know this is unsave, but the other options do not work :) Deal with it!
            struc = eval(ds.attrs['Structure'])
        return att, struc
    
    """ Default data extraction, makes a dictionary in which the data is structured the same as the the given structure """
    def extract_data(self,fn):
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


class DataHandler_Single(DataHandler_Base):
    """ This dataset handler processes X, Y, Z Configuration. it passes the index of the data so we are able to split the data into seperate parts"""
    def __init__(self, filename, split=(75, 12.5, 12.5), seed=None, batchsize=256, conditions=None, input_shape=512,X_key="X") -> None:
        super().__init__(filename, split, seed, batchsize, conditions)
        self.input_shape = input_shape
        self.X_key = X_key 

    def get_dataset(self,idx):
        ds  = tf.data.Dataset.from_tensor_slices(((idx,self.data[self.X_key][idx,:self.input_shape]),self.data["Y"][idx]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.shuffle(ds.cardinality())
        return ds.batch(self.batchsize)
    
    
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