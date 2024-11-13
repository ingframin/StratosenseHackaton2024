from sdr import *
from time import time
from datetime import datetime as dt
import json
from threading import Thread
import numpy as np
from .utilities import *

class Sensor:
    def __init__(self,sdr,samp_rate,chunk_size) -> None:
        self.sdr = sdr
        self.sdr.set_sample_rate(samp_rate)
        self.sdr.change_buffer_size(chunk_size)
        self.stop = False

    def measure(self, duration, freqs, fg, filename ='result.txt', thr=False):
        self.freqs = freqs
        self.fg = fg
        self.duration = duration
        self.result_file = filename
        self.stop = False    
        if thr: 
            self.tr = Thread(target=self._run)
            self.tr.start()
        else:
            self._run()
        
    def halt(self):
        self.stop = True
        self.tr.join()

    def _run(self,add_gain=0):
        start_t = time()
        while time()-start_t < self.duration and not self.stop:
            
            for band in self.freqs:
                
                    P = 0
                    num = 0
                    
                    for fval in np.arange(band['F0']+band['stp']/2,band['F1']+band['stp']/2,band['stp']):
                        F = self.sdr.set_frequency(fval,band=self.sdr.get_sample_rate())
                        gset = find_gain(self.fg,fval)+add_gain
            
                        G = self.sdr.set_gain(abs(gset))
                        #time.sleep(0.001)
                        self.sdr.read_samples()
                        pow = self.sdr.received_power_dB()

                        P += pow
                        num += 1                        
                    
                    res = {'G':G,'Band':band['name'],'F0':band['F0'],'F1':band['F1'], 'P':P/num,'T':str(dt.now()),'Operator':band['operator'],'type':'Full'}
                    print(res)
        
                    with open(self.result_file,'a') as res_file:
                        print(json.dumps(res),file=res_file)




