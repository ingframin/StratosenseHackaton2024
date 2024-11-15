import adi
import numpy as np
import h5py
from time import sleep

def connect_radio(addr='ip:pluto.local',buf_size=int(2**20),gain=33.0,frequency=2422.0,bandwidth=12.0,sample_rate=12.0):
    radio = adi.Pluto(addr)
    radio.gain_control_mode_chan0 = 'manual'
    radio.rx_buffer_size = buf_size    
    radio.gain_control_mode_chan0 = 'manual'
    radio.rx_hardwaregain_chan0 = gain
    radio.rx_lo = int(frequency*1e6)
    radio.rx_rf_bandwidth = int(bandwidth*1e6)
    radio.sample_rate = int(sample_rate*1e6)
    return radio

def capture_radio(radio,filename='2437.npy',num_iterations=10)->None:
    # For sample rates of max 5MHz, the buffers read are contiguous.
    with open(filename, 'wb') as f:
        readings = []
        for it in range(num_iterations):
            print(f"Start Reading: {it}")
            readings.append(radio.rx())
            
        for it,r in enumerate(readings):
            np.save(f,r)
            


radio = connect_radio(frequency=1090)
capture_radio(radio,filename='ads-b.npy')
