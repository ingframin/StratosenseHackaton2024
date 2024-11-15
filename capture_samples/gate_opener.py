from sdr import sdr,pluto
import numpy as np

num_samps = int(2**18)
num_readings = 100

radio = pluto.Pluto_SDR(con_str='ip:pluto.local',buf_size=num_samps,tx_buf_size=num_samps)
radio.set_bandwidth(0.5)
radio.set_gain(50)
radio.set_frequency(433.96)

def record_samples(radio):

    samples = np.zeros((1,num_samps*num_readings),dtype=np.complex64)
    for it in range(0,num_readings*num_samps,num_samps):
        print(f"Start Reading: {it//num_samps}")
        radio.receive()
        samples[0][it:it+num_samps] = radio.read_samples()

    with open("gate_opener.npy", 'wb') as f:
        np.save(f,samples)


def open_gate(radio):
    samples = np.load('gate_opener.py')
    radio.write_samples(samples)
    radio.transmit()

open_gate(radio)