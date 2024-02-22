import numpy as np
import matplotlib.pyplot as plt
import math
'''
2023.05.12
preamble, signal, data field 모두 생성
but, data field에서 i = 0 일때만 일치하고
그 다음 데이터는 IFFT 결과가 일치 X
이 부분만 수정하면 완료
'''
## Short training
NLSTF = 64
idxLSTF = np.array([-24,-20,-16,-12,-8,-4,4,8,12,16,20,24])
fftLSTF = np.zeros(NLSTF, np.complex128)

fftLSTF[idxLSTF] = 1.472*np.array([1,-1,1,-1,-1,1,-1,-1,1,1,1,1])*(1+1j)
LSTF = np.fft.ifft(fftLSTF)
LSTF = np.hstack((LSTF[-32:],np.tile(LSTF,3)))[:161]
LSTF[0]/=2; LSTF[-1]/=2
#print(np.round(LSTF,3))

## Long training
NLLTF = 64
idxLLTF = np.arange(-26,27)
fftLLTF = np.zeros(NLLTF, np.complex128)
fftLLTF[idxLLTF]=np.array([1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1])

#plt.plot(np.real(fftLLTF),'bx')
#plt.show()

LLTF = np.fft.ifft(fftLLTF)

#print(np.round(LLTF,3))
#plt.plot(np.real(LLTF))
#plt.plot(np.imag(LLTF),'--')
#plt.show()

LLTF = np.hstack((LLTF[-32:],np.tile(LLTF,3)))[:161]
LLTF[0]/=2; LLTF[-1]/=2
#plt.plot(np.real(LLTF))
#plt.plot(np.imag(LLTF),'--')
#plt.show()

## Signal
#conv encode
#print('convolution encoder')

bits = [1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
en_bits = [0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#171_1.111.001
#133_1.011.011
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

#plt.subplot(1,2,1)
#plt.plot(output,'r')
#plt.subplot(1,2,2)
#plt.plot(conv_encode)
#plt.show()

##interleaving
#print('interleaving')

interleaved =  []

N_CBPS=48
for idx in range (0,N_CBPS):
    i=idx//3+16*(idx%3)
    interleaved.append(conv_encode[int(i)])

#print(interleaved)

##frequency domain SIGNAL field
lensig=64
SIG = np.zeros(lensig, np.complex128)

sig_sub = np.array([-26,-25,-24,-23,-22,  -20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,  -6,-5,-4,-3,-2,-1,
1,2,3,4,5,6,  8,9,10,11,12,13,14,15,16,17,18,19,20,  22,23,24,25,26])
idxpilot = np.array([-21,-7,7,21])

bi_interleaved = []

for i in range (0,len(sig_sub)):
    #print(i)
    bi_interleaved.append(2*interleaved[i]-1)

SIG[sig_sub] = bi_interleaved
SIG[idxpilot] = np.array([1,1,1,-1])

#print(bi_interleaved)
#print(SIG)

##time domain SIGNAL field
LSIG = np.fft.ifft(SIG)
LSIG = np.hstack((LSIG[-16:],np.tile(LSIG,2)))[:81]
LSIG[0]/=2; LSIG[-1]/=2

#print(np.round(LSIG,3))

##
## Data Adding
L = np.hstack((LSTF[:-1],LSTF[-1]+LLTF[0],LLTF[1:-1],LLTF[-1]+LSIG[0],LSIG[1:-1]))
##

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

        ## Scramblingasdfassdfasdf
        bin_full_scram = scrm_144^bin_full_data


    if(i==0):
        result = bin_full_scram
    else:
        result = np.hstack((result, bin_full_scram))

## Insert Shortening Bits
insert_shortening = np.concatenate((result, np.zeros(642, int)))


## Base Matrix
P = np.array([[48,29,28,39, 9,61,-1,-1,-1,63,45,80,-1,-1,-1,37,32,22, 1, 0,-1,-1,-1,-1],
            [ 4,49,42,48,11,30,-1,-1,-1,49,17,41,37,15,-1,54,-1,-1,-1, 0, 0,-1,-1,-1],
            [35,76,78,51,37,35,21,-1,17,64,-1,-1,-1,59, 7,-1,-1,32,-1,-1, 0, 0,-1,-1],
            [ 9,65,44, 9,54,56,73,34,42,-1,-1,-1,35,-1,-1,-1,46,39, 0,-1,-1, 0, 0,-1],
            [ 3,62, 7,80,68,26,-1,80,55,-1,36,-1,26,-1, 9,-1,72,-1,-1,-1,-1,-1, 0, 0],
            [26,75,33,21,69,59, 3,38,-1,-1,-1,35,-1,62,36,26,-1,-1, 1,-1,-1,-1,-1, 0]], dtype = int)




def generate_ldpc_I(m, k):
    bI = I = np.eye(k, dtype=int)
    I = np.hstack((bI[:,-m:],bI[:,:-m]))

    return I


def generate_ldpc_h(k):

    for i in range(len(P)):
        for j in range(len(P.T)):
            if(P[i, j] > 0):
                I = generate_ldpc_I(P[i, j], k)
            elif (P[i, j] == 0):
                I = np.eye(k,dtype=int)
            else :
                I = np.zeros((k, k), dtype=int)


            if(j==0):
                Hj = I
            else:
                Hj = np.hstack((Hj, I))
        if(i==0):
            H = Hj
        else:
            H = np.vstack((H, Hj))

    return H

def generate_ldpc_G(n, k, H):
    G = np.eye(k, dtype=int)
    p = H[:k, :k];
    G = np.hstack((G, p.T))

    return G


import numpy as np


##
subblock = 81

H = generate_ldpc_h(subblock)

n = len(H.T)
k = n-len(H)

G = generate_ldpc_G(n, k, H)
#G = H.T

codeword = np.dot(insert_shortening, G) % 2
print(codeword)




        ##

        # ### De-scrambler
        # de_scram = []
        #
        # ## convolution encoder PART_1.
        # R=3/4
        # #171_1.111.001
        # #133_1.011.011
        # encode_1 = np.array([1,0,1,1,0,1,1])
        # encode_2 = np.array([1,1,1,1,0,0,1])
        #
        # get = []
        # output = []
        # conv_encode = []
        # c_1=0
        # c_2=0
        # n = 1
        #
        # z = np.zeros(6,int)
        # s = np.hstack((z,bin_full_scram,z))
        # N=len(s)
        #
        # for idx in range(6,N-6):
        #     # if(idx+1 > N+6*(len(bin_full_data)/144)):
        #     #     break
        #     # if(idx == n*143+6):
        #     #     s = np.hstack((z,s[n*143+6:]))
        #     #     n = n + 1
        #     for l in range(0,7):
        #         get.append(s[idx-l])
        #     for j in range(0,7):
        #         c_1+=(get[j]*encode_1[j])
        #         c_2+=(get[j]*encode_2[j])
        #     output.append(c_1)
        #     output.append(c_2)
        #     c_1=0; c_2=0
        #     get.clear()
        #
        # for k in output:
        #     if(k==0):
        #         conv_encode.append(0)
        #     else:
        #         conv_encode.append(k%2)
        #
        # #print("conv_encode")
        # #print(conv_encode)
        #
        # ## convolution encoder PART_2.
        # count = 0
        # conv_result = []
        #
        # for idx in range(0,len(conv_encode)):
        #     if(idx == 3+count*6):
        #         continue
        #     elif(idx-1 == 3+count*6):
        #         count=count+1
        #         continue
        #     else:
        #         conv_result.append(conv_encode[idx])
        #
        # #print("conv_result")
        # #print(conv_result)
        #
        #
        # ## interleaving PART_1.
        # interleaved=np.zeros(len(conv_result),int)
        #
        # for idx in range (0,len(conv_result)):
        #     x=idx//16+12*(idx%16)
        #     #print(x)
        #     interleaved[x]=conv_result[int(idx)]
        #
        # ## interleaving PART_2.
        # interleave_result=np.zeros(len(interleaved),int)
        #
        # for idx in range (0,len(interleaved)):
        #     if(idx%24 <= 11):
        #         interleave_result[idx]=interleaved[idx]
        #         #print(idx)
        #     elif(idx%2 == 0):
        #         interleave_result[idx+1]=interleaved[idx]
        #         #print(idx+1)
        #     else:
        #         interleave_result[idx-1]=interleaved[idx]
        #         #print(idx-1)
        #
        # ### viterbi decoder
        #
        # ## 16_QAM
        # # pilot
        # pilot = np.array([1,1,1,-1])*(1+0j)
        #
        # pi = [1,1,1,  -1,-1,-1,1,  -1,-1,-1,-1,  1,1,-1,1,  -1,-1,1,1,  -1,1,1,-1,  1,1,1,1,  1,1,-1,1,
        #     1,1,-1,1,  1,-1,-1,1,  1,1,-1,1,  -1,-1,-1,1,  -1,1,-1,-1,  1,-1,-1,1,  1,1,1,1,  -1,-1,1,1,
        #     -1,-1,1,-1,  1,-1,1,1,  -1,-1,-1,1,  1,-1,-1,-1,  -1,1,-1,-1,  1,-1,1,1,  1,1,-1,1,  -1,1,-1,1,
        #     -1,-1,-1,-1,  -1,1,-1,1,  1,-1,1,-1,  1,1,1,-1,  -1,1,-1,-1,  -1,1,1,1,  -1,-1,-1,-1,  -1,-1,-1
        #     ]#앞에 signal filed의 pilot 1 빼고 시작
        #
        # # Mapping
        # M = 4           # M=4 for 16-QAM
        # mapping_table = {       # gray code mapping
        #     (0,0,0,0) : -3-3j,
        #     (0,0,0,1) : -3-1j,
        #     (0,0,1,0) : -3+3j,
        #     (0,0,1,1) : -3+1j,
        #     (0,1,0,0) : -1-3j,
        #     (0,1,0,1) : -1-1j,
        #     (0,1,1,0) : -1+3j,
        #     (0,1,1,1) : -1+1j,
        #     (1,0,0,0) :  3-3j,
        #     (1,0,0,1) :  3-1j,
        #     (1,0,1,0) :  3+3j,
        #     (1,0,1,1) :  3+1j,
        #     (1,1,0,0) :  1-3j,
        #     (1,1,0,1) :  1-1j,
        #     (1,1,1,0) :  1+3j,
        #     (1,1,1,1) :  1+1j
        # }
        #
        # #gen 16_QAM
        # ofdm = 64
        # DATA = np.zeros(ofdm, np.complex128)
        #
        # interleave_result = np.array(interleave_result).reshape(-1, 4)
        # output_data = (1/math.sqrt(10))*np.array([mapping_table[tuple(row)] for row in interleave_result])
        #
        # sc_data=np.array([-26,-25,-24,-23,-22, -20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8, -6,-5,-4,-3,-2,-1,
        #                     1,2,3,4,5,6,  8,9,10,11,12,13,14,15,16,17,18,19,20,  22,23,24,25,26])
        # sc_pilot = np.array([-21,-7,7,21])
        #
        # DATA[sc_data] = output_data
        # DATA[sc_pilot] = pi[i]*pilot
        # #FINAL_DATA.append(DATA)
        #
        # ## IFFT
        # #print(np.round(DATA,3))
        # #print(len(DATA))
        # FINAL_IFFT = np.fft.ifft(DATA)
        # FINAL_IFFT = np.hstack((FINAL_IFFT[-16:],np.tile(FINAL_IFFT,2)))[:81]
        # FINAL_IFFT[0]/=2; FINAL_IFFT[-1]/=2
        # #print(np.round(FINAL_IFFT,3))
        #
        # ##Data adding
        # if(i==0):
        #     print(len(FINAL_DATA))
        #     FINAL_DATA=np.hstack((L,LSIG[-1]+FINAL_IFFT[0],FINAL_IFFT[1:]))
        # else:
        #     print(len(FINAL_DATA))
        #     FINAL_DATA=(np.hstack((FINAL_DATA[:-1],FINAL_DATA[-1]+FINAL_IFFT[0],FINAL_IFFT[1:])))
        #


















