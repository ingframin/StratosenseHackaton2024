import adi
from .sdr import *
import numpy as np


class Pluto_SDR(SDR):
    def __init__(self,con_str:str, buf_size: int, tx_buf_size:int=1024):
        super().__init__(buf_size)
        self.sdr = adi.Pluto(con_str)
        self.sdr.gain_control_mode_chan0 = 'manual'
        self.sdr.rx_buffer_size = buf_size
        self.sdr.tx_buffer_size = tx_buf_size
        self.rx_buffer:np.ndarray = np.ndarray([buf_size],dtype=np.complex64)
        self.tx_buffer:np.ndarray = np.ndarray([tx_buf_size],dtype=np.complex64)
        self.valid_gains:list[float] = [x/10 for x in range(0,750,5)]
    
    def set_gain(self, gain:float)->float:
        """Sets the gain in dB"""
        self.set_rx_gain(gain)
        return float(self.sdr.rx_hardwaregain_chan0)
    
    def set_rx_gain(self, gain:float)->None:
        """Sets the receiver gain in dB"""
        self.sdr.rx_hardwaregain_chan0 = gain
        
    
    def set_tx_gain(self, gain)->None:
        """Sets the transmitter gain in dB"""
        self.sdr.tx_hardwaregain_chan0 = gain
    
    def set_frequency(self, freq:float)->float:
        self.sdr.rx_lo = int(freq*1e6)
        self.sdr.tx_lo = int(freq*1e6)
        return float(self.sdr.rx_lo)
    
    def set_rx_frequency(self, freq:int)->int:
        """Sets the receiver centre frequency in Hz"""
        self.sdr.rx_lo = freq
        return self.sdr.rx_lo
    
    def set_tx_frequency(self, freq:int)->int:
        """Sets the transmitter centre frequency in Hz"""
        self.sdr.tx_lo = freq
        return self.sdr.rx_lo
    
    # Separated bandiwdth and frequency setting
    def set_bandwidth(self, band:float)->int:
        self.sdr.rx_rf_bandwidth = int(band*1e6)
        self.sdr.tx_rf_bandwidth = int(band*1e6)
        return self.sdr.rx_rf_bandwidth
    
    def set_rx_bandwidth(self, band:int)->int:
        """Sets the receiver bandwidth in Hz"""
        self.sdr.rx_rf_bandwidth = band
        return self.sdr.rx_rf_bandwidth
    
    def set_tx_bandwidth(self, band:int)->int:
        """Sets the transmitter bandwidth in Hz"""
        self.sdr.tx_rf_bandwidth = band
        return self.sdr.tx_rf_bandwidth

    def read_samples(self)->np.ndarray:
        """Returns a shallow copy of the receive buffer"""
        return self.rx_buffer
    
    def write_samples(self, samples:np.ndarray)->None:
        """Fills the transmite buffer with samples. The "sample" parameter should be the same size as the transmit buffer"""
        self.tx_buffer[:] = samples
         
    def receive(self)->None:
        """Fills the receive buffer with samples coming from the Pluto SDR"""
        self.rx_buffer = self.sdr.rx()

    def transmit(self)->None:
        """Transmits the samples in the Tx buffer"""
        self.sdr.tx(self.tx_buffer)

    def set_sample_rate(self,samp_rate:float)->float:
        super().set_sample_rate(samp_rate)
        self.sdr.sample_rate = int(samp_rate*1e6)
        return self.sdr.sample_rate
    
    def set_sample_rate_Hz(self,samp_rate:int)->int:
        super().set_sample_rate(samp_rate)
        self.sdr.sample_rate = samp_rate
        return self.sdr.sample_rate

    def change_buffer_size(self,new_size:int)->int:
        # We should foresee a MAX_BUFFER_SIZE constant for each SDR
        # To be seen...
        bfs = super().change_buffer_size(new_size)
        self.sdr.rx_buffer_size = len(self.buffer)
        return bfs 
    
    def change_rx_buffer_size(self,new_size:int)->None:
        # We should foresee a MAX_BUFFER_SIZE constant for each SDR
        # To be seen...
        bfs = super().change_buffer_size(new_size)
        self.sdr.rx_buffer_size = new_size
        self.rx_buffer = np.ndarray([new_size],dtype=np.complex64)

    def change_tx_buffer_size(self,new_size:int)->None:
        self.sdr.tx_buffer_size = new_size
        self.tx_buffer = np.ndarray([new_size],dtype=np.complex64)
        
    def get_current_frequency(self)->float:
        return float(self.sdr.rx_lo)
    
    def get_current_frequency_Hz(self)->int:
        return self.sdr.rx_lo
    
    def get_current_gain(self)->float:
        return self.sdr._get_iio_attr('voltage0','hardwaregain', False)

    def get_valid_gains(self):
        return self.valid_gains
