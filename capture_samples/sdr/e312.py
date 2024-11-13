from .sdr import *
import uhd
import numpy as np

class ETTUS_E312(SDR):
    def __init__(self, buf_size:int,recv_buffer_size:int = 2**18):
        super().__init__(buf_size)
        self.sdr = uhd.usrp.MultiUSRP()
        self.recv_buffer_size = min(recv_buffer_size,buf_size)

        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = [0]
        self.metadata = uhd.types.RXMetadata()
        self.streamer = self.sdr.get_rx_stream(st_args)
        self.recv_buffer = np.zeros((1, self.recv_buffer_size), dtype=np.complex64)

        # Start Stream
        self.stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        self.stream_cmd.stream_now = True
        self.stream_cmd.num_samps = len(self.buffer)
        
        
                
    def set_gain(self, gain:float)->float:
        self.sdr.set_rx_agc(False, 0)
        if gain <0:
            gain=0
        self.sdr.set_rx_gain(gain, 0)
        return self.sdr.get_rx_gain(0)
    
    def set_frequency(self, freq:float, band:float=2.4)->float:
        self.sdr.set_rx_freq(uhd.libpyuhd.types.tune_request(freq*1e6), 0)
        self.sdr.set_rx_bandwidth(band*1e6, 0)
        return self.sdr.get_rx_freq(0)
        
    def read_samples(self)->np.ndarray:
        num_samps = len(self.buffer)
        self.streamer.issue_stream_cmd(self.stream_cmd)

        # Receive Samples
        for i in range(num_samps//self.recv_buffer_size):
            hmm = self.streamer.recv(self.recv_buffer, self.metadata)
            self.buffer[i*self.recv_buffer_size:(i+1)*self.recv_buffer_size] = self.recv_buffer[0]
        return self.buffer

    
    def set_sample_rate(self,samp_rate:float)->float:
        self.sdr.set_rx_rate(samp_rate*1e6, 0)
        return self.sdr.get_rx_rate(0)

    def get_current_frequency(self):
        return self.sdr.get_rx_freq(0)
        
    def get_current_gain(self):
        return self.sdr.get_rx_gain(0)
    
    def get_current_sample_rate(self):
        return self.sdr.get_rx_rate(0)

   

    