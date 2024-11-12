from modulations import *
from random import randint, choice, random
import numpy as np
from channel import *
import shelve

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
    

             
messages = [Message(fsk_modulate) for _ in range(32)]
delays = [0.1*random() for _ in range(512)]
attenuations = [random() for _ in range(512)]

for m in messages:
    m.compute_variations(delays,attenuations)

with shelve.open('fsk_data') as dataset:
    dataset["messages"] = messages

