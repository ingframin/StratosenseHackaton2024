from abc import ABC,abstractmethod
import numpy as np
from numpy import log10

from math import log2,ceil

# Sampling frequency is in MHz
def compute_buf_size(time:float,samp_frequency:float):
    '''Returns the closest power of 2 to the product between time and sampling rate in MHz'''
    buf_size = int(time*samp_frequency*1e6)
    if buf_size & (buf_size-1):
        buf_size = 1<<(ceil(log2(buf_size)))
    return buf_size

# Calculates how many iteration to obtain the closest approximaion
# of the desired time depending on the chunk size
# Sampling frequency is in MHz
def iters(time_s:float, size:int, samp_frequency:float)->int:
    Tchunk = size/(samp_frequency*1E6)
    return ceil(time_s/Tchunk)

class SDR(ABC):
    def __init__(self,buf_size:int):
        """buf_size should be a power of 2. It can be calculated with the function compute_buf_size()"""
        self.buffer:np.ndarray = np.ndarray([buf_size],dtype=np.complex64)
        self.Zin = 50
        self.__sample_rate = 0.0

    @abstractmethod
    def set_gain(self, gain:float)->float:
        """Sets the gain in dB"""
        pass

    @abstractmethod
    def set_frequency(self, freq:float)->float:
        """Sets the frequency in MHz."""
        pass

    @abstractmethod
    def read_samples(self)->np.ndarray:
        """Reads samples in the buffer and returns a pointer to the buffer"""
        pass
    
    @abstractmethod
    def set_sample_rate(self,samp_rate:float)->float:
        """Sets the sample rate in MHz"""
        self.__sample_rate = samp_rate
        return self.__sample_rate
        

    @abstractmethod
    def get_current_frequency(self):
        pass
    
    @abstractmethod
    def get_current_gain(self):
        pass
    
    def get_sample_rate(self)->float:
        """returns the current sample rate"""
        return self.__sample_rate

    def rms(self)->float:
        """Returns the RMS of the values in the buffer"""
        return np.sqrt(np.average(np.square(np.abs(self.buffer))))

    def received_power(self, N: float=1.0)->float:
        """Calculates the received power over the sampling time in W. N is a correction factor to account for gain losses in the radio frontend"""
        return N*(self.rms()**2)/self.Zin 

    def received_power_dB(self, N: float=1.0)->float:
        """Calculates the received power over the buffer in dB"""
        return 10*log10(self.received_power(N))

    def change_buffer_size(self,new_size:int)->int:
        buf_size = 1<<(ceil(log2(new_size)))
        self.buffer = np.ndarray([buf_size],dtype=np.complex64)
        return buf_size