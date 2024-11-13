import uhd
import numpy as np

usrp = uhd.usrp.MultiUSRP()

num_samps = 2**24 # number of samples received
center_freq = 2462e6 # Hz
sample_rate = 12e6 # Hz
gain = 50 # dB

usrp.set_rx_rate(sample_rate, 0)
usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
usrp.set_rx_gain(gain, 0)

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
with open("2462_usrp.npy", 'wb') as f:
    readings = np.zeros((10,num_samps))
    for it in range(10):
        print(f"Start Reading: {it}")
        streamer.recv(recv_buffer, metadata)
        readings[it][:] = recv_buffer[0]
        
    for it,r in enumerate(readings):
        np.save(f,r)

# Receive Samples
samples = recv_buffer[0]

# Stop Stream
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
streamer.issue_stream_cmd(stream_cmd)

