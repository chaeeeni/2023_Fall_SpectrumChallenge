import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.io import loadmat
'''
2023.06. 02
preamble, signal, data field 모두 생성
ldpc encoding 완성. coderate = 3/4, datanum = 1944
'''
## Short training
NLSTF = 64
idxLSTF = np.array([-24,-20,-16,-12,-8,-4,4,8,12,16,20,24])
fftLSTF = np.zeros(NLSTF, np.complex128)

fftLSTF[idxLSTF] = 1.472*np.array([1,-1,1,-1,-1,1,-1,-1,1,1,1,1])*(1+1j)
LSTF = np.fft.ifft(fftLSTF)
LSTF = np.hstack((LSTF[-32:],np.tile(LSTF,3)))[:161]
LSTF[0]/=2; LSTF[-1]/=2

## Long training
NLLTF = 64
idxLLTF = np.arange(-26,27)
fftLLTF = np.zeros(NLLTF, np.complex128)
fftLLTF[idxLLTF]=np.array([1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1])

LLTF = np.fft.ifft(fftLLTF)

LLTF = np.hstack((LLTF[-32:],np.tile(LLTF,3)))[:161]
LLTF[0]/=2; LLTF[-1]/=2

## Signal
bits = [1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
en_bits = [0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

encode_1 = np.array([1,0,1,1,0,1,1])
encode_2 = np.array([1,1,1,1,0,0,1])
get=[]
output=[]
conv_encode=[]
c_1=0
c_2=0

for idx in range(6,len(bits)+6):
    for i in range(0,7):
        get.append(en_bits[idx-i])
    for j in range(0,7):
        c_1+=(get[j]*encode_1[j])
        c_2+=(get[j]*encode_2[j])
    output.append(c_1)
    output.append(c_2)
    c_1=0; c_2=0
    get.clear()

for k in output:
    if(k==0):
        conv_encode.append(0)
    else:
        conv_encode.append(k%2)

##interleaving
interleaved =  []

N_CBPS=48
for idx in range (0,N_CBPS):
    i=idx//3+16*(idx%3)
    interleaved.append(conv_encode[int(i)])

##frequency domain SIGNAL field
lensig=64
SIG = np.zeros(lensig, np.complex128)

sig_sub = np.array([-26,-25,-24,-23,-22,  -20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,  -6,-5,-4,-3,-2,-1,
1,2,3,4,5,6,  8,9,10,11,12,13,14,15,16,17,18,19,20,  22,23,24,25,26])
idxpilot = np.array([-21,-7,7,21])

bi_interleaved = []

for i in range (0,len(sig_sub)):
    bi_interleaved.append(2*interleaved[i]-1)

SIG[sig_sub] = bi_interleaved
SIG[idxpilot] = np.array([1,1,1,-1])

##time domain SIGNAL field
LSIG = np.fft.ifft(SIG)
LSIG = np.hstack((LSIG[-16:],np.tile(LSIG,2)))[:81]
LSIG[0]/=2; LSIG[-1]/=2

## Data Adding
L = np.hstack((LSTF[:-1],LSTF[-1]+LLTF[0],LLTF[1:-1],LLTF[-1]+LSIG[0],LSIG[1:-1]))

## data bits (octets -> bits)
mac_header = np.array([0x04,0x02,0x00,0x2e,0x00,
                       0x60,0x08,0xcd,0x37,0xa6,
                       0x00,0x20,0xd6,0x01,0x3c,
                       0xf1,0x00,0x60,0x08,0xad,
                       0x3b,0xaf,0x00,0x00])
a='''Joy, bright spark of divinity,
Daughter of Elysium,
Fire-insired we trea'''

message = np.zeros(72,dtype=np.int32)
for idx in range(len(a)):
    message[idx]=ord(a[idx])

crc32 = np.array([0x67,0x33,0x21,0xb6])
data_hex = np.hstack((mac_header,message,crc32))
data_bin = np.array([format(c,'08b') for c in data_hex])

little_en=[]
for idx in range(len(data_bin)):
    bit_list = list(data_bin[idx])
    bit_list.reverse()
    str = ','.join(bit_list)
    little_en.append(str)

result = []
for s in little_en:
    result.extend([int(c) for c in s.split(',')])

## service, tail, pad
service_bit = np.zeros(16,dtype=np.int32)

mb_per_sec = 36
matchnum = math.ceil(822/144)*144
tail_pad_bit = np.zeros(matchnum-len(result)-len(service_bit),dtype=np.int32)
full_data = np.hstack((service_bit,result,tail_pad_bit))

#full_data = np.hstack((service_bit,result))

##devide 144 bits
full_data = np.array(full_data).reshape(-1, 144)

## scrambling123
reg = np.array([1,0,1,1,1,0,1])
N=127
scrm=np.zeros(N, int)
for idx in range(127):
    temp = reg[6]^reg[3]
    reg[1:]=reg[:6]
    reg[0]=temp
    scrm[idx]=temp
scrm = np.tile(scrm,10)[:864]
scrm = np.array(scrm).reshape(-1, 144)


## ## FOR
FINAL_DATA = []
for i in range (0,len(full_data)):
    print(i)
    bin_full_data = full_data[i]
    scrm_144 = scrm[i]

    if(i==len(full_data)-1):
        bin_full_data = bin_full_data[:96]
        scrm_95 = scrm_144[:96]
        bin_full_scram = scrm_95^bin_full_data

    else:
## Scrambling
        bin_full_scram = scrm_144^bin_full_data

    if(i==0):
        result = bin_full_scram
    else:
        result = np.hstack((result, bin_full_scram))

## Insert Shortening Bits
insert_shortening = np.concatenate((result, np.zeros(642, int)))


## main
subblock = 81

matfile = loadmat("PT_3_4_1944.mat")
G = matfile["PT_3_4_1944"]

paritybits = np.dot(insert_shortening, G.T) % 2

codeword = np.hstack((insert_shortening, paritybits))
print(codeword)

## Removing Shortening bits and pucturing for LDPC
codeword = np.hstack((codeword[:816], codeword[1458:1890]))
