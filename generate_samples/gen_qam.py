import numpy as np
from random import randint, choice
import numpy as np
from modulations import *

qam256 = QAM(256)

messages = [[randint(0,255) for _ in range(32)] for _ in range(8)]

result = [[QAM_modulate(byte,qam256) for byte in m] for m in messages]

for _ in range(256):

    result.append(choice(messages))
    result.append(np.zeros(randint(16,128)))

final = np.concatenate(result)

np.save('QAM256_data.npy',final)

