from modulations import *
from random import randint, choice
import numpy as np

messages = [np.array([randint(0,255) for _ in range(32)]) for _ in range(8)]
result = [fsk_modulate(m) for m in messages]


result = []

for _ in range(256):

    result.append(choice(messages))
    result.append(np.zeros(randint(16,128)))

final = np.concatenate(result)

np.save('fsk_data.npy',final)