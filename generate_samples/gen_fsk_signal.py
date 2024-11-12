from modulations import *
from random import randint, choice, random
import numpy as np
from channel import *
import h5py


messages = (np.array([randint(0,255) for _ in range(16)]) for _ in range(32))             
modulated_messages = (fsk_modulate(m,f1=5,f2=10) for id,m in enumerate(messages))
delays = [0.1*random() for _ in range(16)]
attenuations = [random() for _ in range(16)]

with h5py.File('fsk_data.h5', 'w') as f:
    for i,m in enumerate(modulated_messages):
        print(f'Message: {i}')
        data = np.zeros((16, len(m)),dtype=np.complex64)
        for a,d,n in zip(attenuations,delays,range(16)):
            data[n] = add_attenuation(add_delay(m,d,433e6),a)
        f.create_dataset(f'{i}',data=data)


    
    
