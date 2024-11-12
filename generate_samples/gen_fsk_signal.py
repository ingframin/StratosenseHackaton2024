from modulations import *
from random import randint, choice, random
import numpy as np
from channel import *
import shelve
import gc

class Message:
    id = 0
    def __init__(self, mod:callable) -> None:
        self.id = Message.id
        Message.id += 1
        self.content = np.array([randint(0,255) for _ in range(512)])
        self.modulated_content = mod(self.content)
        self.variations = []
    
    def compute_variations(self,delays,attenuations,f=433.9e6):
        for d,a in zip(delays,attenuations):
            self.variations.append(add_delay(add_attenuation(self.modulated_content,a),d,f))
    

             
delays = [0.1*random() for _ in range(16)]
attenuations = [random() for _ in range(16)]

with shelve.open("fsk_dataset") as dataset:
    for i in range(1000):
        m = Message(fsk_modulate)
        print("Message: ",m.id)
        m.compute_variations(delays,attenuations)
        print("variations: ",len(m.variations))
        dataset[str(m.id)] = m
        gc.collect()




