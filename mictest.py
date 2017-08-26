import pyaudio
from struct import pack, unpack
import math
import wave
import numpy as np

INITIAL_TAP_THRESHOLD = 0.010
FORMAT = pyaudio.paInt16 
SHORT_NORMALIZE = (1.0/32768.0)
CHANNELS = 2
RATE = 44100  
INPUT_BLOCK_TIME = 5
INPUT_FRAMES_PER_BLOCK = int(RATE*INPUT_BLOCK_TIME)

pa = pyaudio.PyAudio()

def find_input_device():
    device_index = None            
    for i in range(pa.get_device_count()):     
        devinfo = pa.get_device_info_by_index(i)   
        print( "Device %d: %s"%(i,devinfo["name"]) )

        for keyword in ["mic","input"]:
            if keyword in devinfo["name"].lower():
                print( "Found an input: device %d - %s"%(i,devinfo["name"]) )
                device_index = i
                return device_index

    if device_index == None:
        print( "No preferred input found; using default input device." )

    return device_index

def open_mic_stream():
    device_index = find_input_device()

    stream = pa.open(format = FORMAT,
                     channels = CHANNELS,
                     rate = RATE,
                     input = True,
                     input_device_index = device_index,
                     frames_per_buffer = INPUT_FRAMES_PER_BLOCK)

    return stream

stream = open_mic_stream()

def unsigned_to_signed(x):
    return x if x < (1 << 16-1) else x - (1 << 16)

def signed_to_unsigned(x):
    return x & 0xffff

def shift(freqs):
    mid_high = len(freqs)//2 + 1
    return np.append(np.append(np.append(np.append([freqs[0]], [freqs[0]]), [freqs[1:mid_high-1]]),  
                               [freqs[mid_high+1:]]), [[freqs[-1]]])
def convert(data, flip=True):
    data = bytearray(data)
 
    data_16 = [data[i] + data[i+1] * 256 for i in range (0,len(data), 2)]
    signed_data = np.clip([unsigned_to_signed(x) for x in data_16], -32767, 32767)

    freqs = np.fft.fft(signed_data)
    
    freqs = shift(freqs)
    
    reconstruct = np.clip([int(x) for x in np.real(np.fft.ifft(freqs))], -32767, 32767)
    unsigned_reconstruct = [signed_to_unsigned(x) for x in reconstruct]
    
    output = b''
    for i in range(0, len(data_16)):
        output += pack('H', data_16[i])
        output += pack('H', unsigned_reconstruct[i])
                
    return output

def listen():
    block = stream.read(INPUT_FRAMES_PER_BLOCK)
    wv = wave.open('testmic.wav', mode='wb')
    wv.setnchannels(CHANNELS)
    wv.setsampwidth(2)
    wv.setframerate(RATE)
    wv.writeframesraw(convert(block))
    
if __name__ == "__main__":
    listen()
