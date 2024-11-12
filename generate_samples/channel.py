import numpy as np
import numpy.typing as npt
C=299792458

def add_delay(signal:npt.ArrayLike, time:float, center_frequency:float)->npt.ArrayLike:
    return signal*np.exp(1j*2*np.pi*center_frequency*time)

def add_attenuation(signal:npt.ArrayLike, attenuation:float):
    return signal*attenuation

def log_distance_Prx(frequency:float, distance:float, Ptx:float):
    return Ptx - 20*np.log10(distance) - 20*np.log10(frequency*1e6) - 20*np.log10(4*np.pi/C);
