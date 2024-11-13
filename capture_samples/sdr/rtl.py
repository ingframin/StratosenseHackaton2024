from rtlsdr import *
from .sdr import *
from time import sleep
try:
    from serial import Serial
except:
    print("PySerial not available: Extension board cannot be used")

class RTL_SDR(SDR):
    def __init__(self, buf_size:int, ext_board_port: str=""):
        super().__init__(buf_size)
        if ext_board_port != "":
            try:
                self.ext_board = Serial(ext_board_port,timeout=30)
            except:
                self.ext_board = None
                print("Extension board not found!")
        self.sdr = RtlSdr()
        
    
    def valid_gains(self)->list[float]:
        return self.sdr.valid_gains_db
    
    def set_gain(self, gain:float)->float:
        self.sdr.set_agc_mode(False)
        if gain <0:
            gain=0
        self.sdr.set_gain(gain)
        return self.sdr.gain
    
    def set_frequency(self, freq:float, band:float=0)->float:
        if self.ext_board is not None:
            if self.ext_board is not None:
                self.ext_board.write(b'\n\r')
                self.ext_board.write(b'convert setup %d\n\r'%(freq*1000))
                sleep(0.1)
                l1 = self.ext_board.readline()
                # print("l1=",l1)
                sleep(0.1)
                l3 = self.ext_board.readline()
                # print("l3=",l3)
                sleep(0.1)
                l2 = self.ext_board.readline()
                # print("l2=",l2)
                for s in l2.split():
                    ss = s.decode()       
                    # print("ss=",ss)    
                    if ss.isdecimal():
                        f = int(ss)*1000
                        # print("f=",f)
                        break
            
        else:
            f = freq*1e6

        self.sdr.set_center_freq(f)
        self.sdr.set_bandwidth(band)
        
        return self.sdr.get_center_freq()
    
    def read_samples(self)->np.ndarray:
        self.buffer[:] = self.sdr.read_samples(len(self.buffer))
        return self.buffer
    
    def set_sample_rate(self,samp_rate:float)->float:
        super().set_sample_rate(samp_rate)
        self.sdr.set_sample_rate(samp_rate*1e6)
        return self.sdr.get_sample_rate()

    
    def get_current_frequency(self):
        return self.sdr.center_freq
    
    def get_current_gain(self):
        return self.sdr.get_gain()