import socket
from .sdr import *

"""
This class in only pretending to be a SDR, It is actually a ethernet interface :)
"""


class ETH0(SDR):
    def __init__(self, buf_size:int, server_port: tuple=('192.168.10.42', 65431)):
        super().__init__(buf_size)
        self.sdr = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server= server_port

    def set_gain(self, gain:float)->float:
        if gain <0:
            gain=0
        cmd = b'gain ' + str(gain).encode()
        self.sdr.sendto(cmd,self.server)
        data, addr = self.sdr.recvfrom(1024)
        return float(data.decode().split()[1])
    
    def set_frequency(self, freq:float, band:float=0)->float:
        cmd = b'freq ' + str(freq).encode()+b' '+ str(band).encode()
        self.sdr.sendto(cmd,self.server)
        data, addr = self.sdr.recvfrom(1024)
        return float(data.decode().split()[1])
        
    
    def read_samples(self)->np.ndarray:
        cmd = b'samples'
        self.sdr.sendto(cmd,self.server)
        data, addr = self.sdr.recvfrom(1024)
        #This is not really conformal to the spec. Samples should be saved in the internal buffer 
        # or power calculation doesn't work
        print(data.decode() + " samples are saved on server")
        return self.buffer
        #What?
        cmd = b'samples '+str(len(self.buffer)).encode()
        self.sdr.sendto(cmd,self.server)
        buffer = []
        data = b''
        while b'-' not in data:
            data, addr = self.sdr.recvfrom(4096)
            buffer.append(data)
            b = b''.join(buffer[0:-1])
        samples = np.frombuffer(b,dtype=np.complex64)
        self.buffer[:] = samples
        return self.buffer
    
    def set_sample_rate(self,samp_rate:float)->float:
        super().set_sample_rate(samp_rate)
        cmd = b'samp_rate ' + str(samp_rate).encode()
        self.sdr.sendto(cmd,self.server)
        data, addr = self.sdr.recvfrom(1024)
        return float(data.decode().split()[1])

    def get_current_frequency(self):
        cmd = b'freq_req '
        self.sdr.sendto(cmd,self.server)
        data, addr = self.sdr.recvfrom(1024)
        return float(data.decode().split()[1])
        
    def get_current_gain(self):
        cmd = b'gain_req '
        self.sdr.sendto(cmd,self.server)
        data, addr = self.sdr.recvfrom(1024)
        return float(data.decode().split()[1])
    
    def rms(self) -> float:
        cmd = b'rms '
        self.sdr.sendto(cmd,self.server)
        data, addr = self.sdr.recvfrom(1024)
        return float(data.decode().split()[1])   
    
    def received_power(self, N: float = 1) -> float:
        cmd = b'recv_pow '+ str(N).encode()
        self.sdr.sendto(cmd,self.server)
        data, addr = self.sdr.recvfrom(1024)
        return float(data.decode().split()[1])
    
    def received_power_dB(self, N: float = 1) -> float:
        cmd = b'recv_pow_dB '+ str(N).encode()
        self.sdr.sendto(cmd,self.server)
        data, addr = self.sdr.recvfrom(1024)
        return float(data.decode().split()[1])

    def change_buffer_size(self, new_size: int) -> None:
        cmd = b'change_buff '+ str(new_size).encode()
        self.sdr.sendto(cmd,self.server)
        data, addr = self.sdr.recvfrom(1024)
        return super().change_buffer_size(new_size)