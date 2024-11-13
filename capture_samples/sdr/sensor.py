from sdr import *
import numpy as np
from .utilities import *
from collections import deque
from dataclasses import dataclass
import pickle as pk
from datetime import datetime as dt

@dataclass
class DataFrame:
    id: int
    sample_rate: float
    frequency: float
    gain: float
    data: np.array
    time: dt
    data_type: str # temporary solution, this should be an instance of an enum class with the units of measurement

class Sensor:
    def __init__(self,sdr,samp_rate,chunk_size,freq_gain_table) -> None:
        self.sdr = sdr
        self.fg = freq_gain_table
        self.sdr.set_sample_rate(samp_rate)
        self.sdr.change_buffer_size(chunk_size)
        self.stop = False
        self.data = deque()
        self.buffer = []
        self.df_num = 0
        self.find_gain = FindGain(freq_gain_table)

    def measure(self, f_start, f_stop, f_step):
        
        for fval in np.arange(f_start+f_step/2,f_stop+f_step/2,f_step):
            gset = self.find_gain(self.fg,fval)
            self.sdr.set_gain(gset)
            
            self.sdr.set_frequency(fval,band=self.sdr.get_sample_rate())
            self.buffer=self.sdr.read_samples()
            
            df = DataFrame(self.df_num,self.sdr.get_sample_rate(),fval,gset,self.buffer,dt.now(),"raw samples")
            
            self.data.append(df)
            self.df_num += 1

    def measure_power(self,f_start,f_stop,f_step):
        for fval in np.arange(f_start+f_step/2,f_stop+f_step/2,f_step):
            gset = self.find_gain(self.fg,fval)
            self.sdr.set_gain(gset)
            
            self.sdr.set_frequency(fval,band=self.sdr.get_sample_rate())
            self.sdr.read_samples()
            self.buffer=self.sdr.received_power_dB()
            df = DataFrame(self.df_num,self.sdr.get_sample_rate(),fval,gset,self.buffer,dt.now(), "power in dB")
            
            self.data.append(df)
            self.df_num += 1
        
    def save_data(self, filename):
        with open(filename,'wb') as res_file:
            pk.dump(self.data,res_file)


            
                    




