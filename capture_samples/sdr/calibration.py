from .signal_generator import SignalGenerator
from .sdr import *
from time import sleep
from .data_model import *
from .utilities import *

def find_scale(sdr:SDR, Ptrg:int = -30, prec:float = 0.1, n_ini:float = 1)->int:
    """ Find the correct correction coefficient for the power readings:
        - sdr: Instance of the SDR to use
        - Ptrg: Target power in dBm
        - prec: target precision such that |P(read) - Ptrg| < prec
        - n_ini: init value for N. Starting from the previously calculated value saves time
        - nsamps: number of samples over which the power is computed
    """
    sdr.read_samples()  
    n = int(n_ini*1e6)
    P = sdr.received_power_dB(N=n)
    
    max_it=1000
    while abs(Ptrg-P) > prec and max_it > 0:
        #binary search of the correct value
        if P > Ptrg:
            n -= (n>>1)
        else:
            n += (n>>1)

        sdr.read_samples()  
        
        P = sdr.received_power_dB(N=n*1e-6)
        
        max_it -= 1
        
    
    return n

def characterize_gain(sdr:SDR, sig_gen:SignalGenerator, freqs:list, gains:list, pows:list, thr:float = 100)->Gval:
    readings = []
    for f in freqs:
        sig_gen.set_frequency(f,'MHz')
        sdr.set_frequency(f)
        for p in pows:
            sig_gen.set_amplitude(p,'dBm') 
            for g in gains:
                sdr.set_gain(g)            
                sdr.read_samples()  
                pw_dB = sdr.received_power_dB()
                print('f:',f,'Psg:',p,'P(dB): ',pw_dB,'G:',g)
                gv = Gval(f,g,p,pw_dB)
                readings.append(gv)
                if pw_dB > thr or pw_dB > p+10:
                    break
                sleep(0.1)

    return readings            
    

def characterize_N(sdr:SDR, sig_gen:SignalGenerator, gain_v_freqs:dict, pows:list)->Nval:
    """ Finds the correct value of N vs Frequency"""
    data = []
    for g,f in gain_v_freqs:
        sig_gen.set_frequency(f,'MHz')
        sdr.set_frequency(f)
        for p in pows:
            sig_gen.set_amplitude(p,'dBm') 
            sdr.set_gain(g)
            N = find_scale(sdr, Ptrg = p)
            result = Nval(f,g,p,N=N)
            data.append(result)
            
    return data


