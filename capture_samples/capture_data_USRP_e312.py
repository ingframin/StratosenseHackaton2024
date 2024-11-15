import uhd
import numpy as np
from math import ceil,log2
from time import sleep
def compute_buf_size(time:float,samp_frequency:float):
    '''Returns the closest power of 2 to the product between time and sampling rate in MHz'''
    buf_size = int(time*samp_frequency*1e6)
    if buf_size & (buf_size-1):
        buf_size = 1<<(ceil(log2(buf_size)))
    return buf_size

usrp = uhd.usrp.MultiUSRP()

# num_samps = compute_buf_size(1.0,12) # number of samples received
num_samps = int(1.2e4)
print(num_samps)
center_freq = 816e6 # Hz
sample_rate = 12e6 # Hz
gain = 70.0 # dB
num_readings = 1000
usrp.set_rx_rate(sample_rate, 0)
usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
usrp.set_rx_gain(gain, 0)
usrp.set_rx_bandwidth(12e6, 0)

# Set up the stream and receive buffer
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
metadata = uhd.types.RXMetadata()
streamer = usrp.get_rx_stream(st_args)
recv_buffer = np.zeros((1, num_samps), dtype=np.complex64)

# Start Stream
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
streamer.issue_stream_cmd(stream_cmd)
with open("movistar_downlink_811-821MHz_usrp_mini_10MHzband_12.6MHzsamp.npy", 'wb') as f:
    readings = np.zeros((1,num_readings*num_samps),dtype=np.complex64)
    for it in range(0,num_readings*num_samps,num_samps):
        print(f"Start Reading: {it//num_samps}")
        streamer.recv(recv_buffer, metadata)
        readings[0][it:it+num_samps] = recv_buffer[0]
        # sleep(1.0)
        
    
    np.save(f,readings)

# Receive Samples
samples = recv_buffer[0]

# Stop Stream
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
streamer.issue_stream_cmd(stream_cmd)

