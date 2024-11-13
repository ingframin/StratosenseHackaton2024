import numpy as np
from .sdr import SDR

def time_avg_power(sdr:SDR, N:float, period:float, nsamps:int, samp_rate:float)->float:
    """ Computes the average power over a period of time:
        - sdr: instance of an SDR object
        - N: correction coefficient
        - period: average period in seconds
        - nsamps: SDR buffer length
        - samp_rate: sample rate of the SDR"""

    steps = int(period*samp_rate*1e6/nsamps)
    power_readings = []
    
    for _ in range(steps):
        sdr.read_samples()
        power_readings.append(sdr.received_power(N))
    
    return sum(power_readings)/len(power_readings)


def elefield(frequency:float, power:float,Gant:float):
    '''Gant and power are in dB, frequency in MHz'''
    P = 10**((power-Gant)/10)
    E = (120*np.pi*frequency/300)*np.sqrt(30*P)
    return E
