from random import randint, choice
import numpy as np
from numpy import typing as npt

def is_power_of_two(num:int)->bool:
    """Check if a number is a power of 2"""
    return num > 0 and np.log2(num).is_integer()


def generate_complex_sine_wave(freq:float, sample_rate:int)->npt.NDArray[np.complex64]:
    """Generate complex sine wave"""
    t = np.arange(sample_rate) / sample_rate
    complex_wave = np.sin(2 * np.pi * freq * t) + 1j * np.cos(2 * np.pi * freq * t)
    return complex_wave.astype(np.complex64)

def OFDM_mod(syms:npt.NDArray[np.complex64])->npt.NDArray[np.complex64]:
    return np.fft.ifft(syms)

# function to demodulate OFDM symbols
def OFDM_demod(ofdm_syms:npt.NDArray[np.complex64])->npt.NDArray[np.complex64]:
    return np.fft.fft(ofdm_syms)

def bytes_to_nibbles(bytes:list[int])->list[int]:
    """Convert bytes to nibbles"""
    nibbles = []
    for b in bytes:
        nibbles.append(b>>4)
        nibbles.append(b&0x0F)
    return nibbles

def nibbles_to_bytes(nibbles:list[int])->list[int]:
    """Convert nibbles to bytes"""
    bytes = []
    for i in range(0,len(nibbles),2):
        bytes.append((nibbles[i]<<4)+nibbles[i+1])
    return bytes

def padding(syms:npt.NDArray[np.complex64], padding:int)->npt.NDArray[np.complex64]:
    """
    Insert padding zeros in between the symbols of a complex signal
    """
    result = []
    for i in range(len(syms)):
        result.append(syms[i])
        result.extend([0.0]*padding)
    #result = np.array(result,dtype=np.complex64) #this is useless if types are correct
    padded_sym = np.array(result,dtype=np.complex64)
    
    return padded_sym

def depadding(syms:npt.NDArray[np.complex64], padding:int)->npt.NDArray[np.complex64]:
    """
    Remove padding zeros in between the symbols of a complex signal
    """
    result = []
    for i in range(0,len(syms),padding+1):
        result.append(syms[i])
    return np.array(result,dtype=np.complex64)

def QAM(N:int)->dict[int,complex]:
    '''Generates a constellation of (sqrt(N)-1)/2 QAM symbols'''
    symbols = []
    k = (np.sqrt(N)-1)/2
    for I in range(int(np.sqrt(N))):
        for Q in range(int(np.sqrt(N))):
            
            symbols.append((I-k)+(Q-k)*1j)
    
    return {n:qc for n,qc in zip(range(N),(1/k)*np.array(symbols,dtype=np.complex64))}

def QAM_modulate(symbol:int, constellation:dict[int,complex])->complex:
    '''Modulates a symbol using a QAM constellation'''
    return constellation[symbol]

def QAM_demodulate(symbol:int, constellation:dict[int,complex])->int:
    '''Demodulates a symbol using a QAM constellation'''
    min = 10
    cur_sym = 0
    for cd in constellation:
        dif = abs(symbol-constellation[cd])
        if dif<min:
            cur_sym = cd
            min = dif
    
    return cur_sym 


def gen_carrier(freq:float,length)->np.ndarray:
    w = np.linspace(-np.pi,np.pi,length)
    C = np.cos(freq*w)
    S = np.sin(freq*w)
    return C+1j*S

def byte2bits(byte,MSB_first=True):
    bits = []
    for i in range(8):
        mask = 1<<i
        if byte&mask:
            bits.append(1)
        else:
            bits.append(0)
    if MSB_first:
        bits.reverse()
    return bits


def fsk_modulate(data:bytearray,f1=1000,f2=2000):
    result = []
    carrier_f1 = gen_carrier(f1,f2*2)
    carrier_f2 = gen_carrier(f2,f2*2)
    for d in data:
        bits = byte2bits(d)
        for b in bits:
            if b:
                result.append(carrier_f1)
            else:
                result.append(carrier_f2)
    return np.concatenate(result)






