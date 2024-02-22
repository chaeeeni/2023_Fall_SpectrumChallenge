'''
Coded by Youngsik Kim @ CSEE . HGU 2020. 07. 25
GMSK Direct implementation #1 - complex modulation / demodulation
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

## 0. setup the environment variable
Nbit=1000
Ts=0.1
fc=2.5  #carrier frequency
Tb=1
Nov=int(Tb/Ts)  # samples per symbol
SNR=10  #SNR in dB

cfo=np.random.uniform(-0.25,0.25)
npx=np.random.randint(100,1000,1) # number of prepand

## 1.0 Genrate pulse shpe / Rect / Sin / Gaussian
# 1. Rectangular
ti=np.arange(0,Tb,Ts)
ps=np.ones(Nov)

# # 2. Guassian
# BT=0.5
# sigma= Tb/2/np.pi*np.sqrt(2*np.log(2))/BT
#
# ti=np.arange(0,2*Tb,Ts)
# ps=1/np.sqrt(2*np.pi)/sigma*np.exp(-(ti-Tb)**2/2/sigma**2)
# ps = ps*2*Nov/np.sum(ps)
# off_set=len(ps)-1

## T1. generate data
bs=np.sign(np.random.normal(size=(Nbit)))

## T2. pulse-shaping
#upsampling and filtering
bsup=np.zeros(len(bs)*Nov)
bsup[0::Nov]=bs
# pulse shaping
tx_data_ps=np.convolve(ps,bsup)

## T3. accumlate
tx_acc = np.add.accumulate(tx_data_ps)/Nov * np.pi/2  # df=1/2Tb, theta = pi/Tb*Ts

## T4. modulate the carrier signal
ts=np.arange(0,len(tx_acc)*Ts,Ts)
tx_data_pass = np.exp(1j*(tx_acc+2*np.pi*fc*ts))


## Channel model ( TX to RX ) carrier offset, time offset, and noise
#cof effect
tx_data_pass = tx_data_pass * np.exp(2j*np.pi*cfo*ts)

# AWGN adding ( SNR control )
nt=np.random.normal(size=len(tx_acc)+npx)+1j*np.random.normal(size=len(tx_acc)+npx)
nt=nt/np.sqrt(2)/10**(SNR/20)
rx_data_pass = np.append(np.zeros(npx),tx_data_pass) +nt


## R1 find the start index

# estimate signal presence by use of RSSI

start_index=len(rx_data_pass)
span=Nov
Prx_th=10
Prx=[]
for idx in range(len(rx_data_pass)-span):
    rx=rx_data_pass[idx:idx+span]
    Prx.append(np.real(rx.dot(rx.conj())))
    if start_index>idx and Prx[idx]>Prx_th:
        start_index=idx
t_prx=np.linspace(0,len(Prx)-1,len(Prx))
rx_data_detected=rx_data_pass[start_index:len(rx_data_pass)]
tsx=np.arange(0,len(rx_data_detected)*Ts,Ts)


## R2 down conversion and correct carrier freqeuncy offset

#carrier offset estimate
theta_rx = np.arctan2(np.imag(rx_data_detected),np.real(rx_data_detected))

#convert to continuous phase
for idx in range(len(theta_rx))[1:]:
    theta_rx[idx]=(theta_rx[idx]-theta_rx[idx-1]+np.pi)%(2*np.pi)-np.pi+theta_rx[idx-1]
fc_hat=(theta_rx[-1]-theta_rx[0])/2/np.pi/len(theta_rx)/Ts
print('Estimated carrier = %.3f',fc_hat)

# down conversion and cfo correction
rx_data_dc = rx_data_detected * np.exp(-2j*np.pi*fc_hat*tsx)

# find the base band phase
rx_data = np.arctan2(np.imag(rx_data_dc),np.real(rx_data_dc))
#convert to continuous phase
for idx in range(len(rx_data))[1:]:
    rx_data[idx]=(rx_data[idx]-rx_data[idx-1]+np.pi)%(2*np.pi)-np.pi+rx_data[idx-1]

## R3 matched filter
rx_data_matched=np.convolve(ps,rx_data)
trx=np.arange(0,len(rx_data_matched)*Ts,Ts)

## R4 differentiation
rx_data_diff=np.zeros(len(rx_data_matched))
rx_data_diff[0]=rx_data_matched[0]
for idx in range(len(rx_data))[1:]:
    rx_data_diff[idx]=rx_data_matched[idx]-rx_data_matched[idx-1]

rx_data_bb = rx_data_diff / np.sqrt(np.var(rx_data_diff))

## R5 Timing Recovery

#generate eye pattern ( 3 eye )
Neye=3
cols=Neye*Nov
rows=len(rx_data_bb)/cols
nzpd=int(np.ceil(rows))*cols - len(rx_data_bb)  # number of zeropadding
rows=int(np.ceil(rows))

shape = (rows,Neye)
ts_eye=np.linspace(0,cols,cols)
rx_data_eye = np.reshape(np.append(rx_data_bb,np.zeros(nzpd)),(rows,cols))

# Estimate Optimze Timing over 5 symbols
fcost=[]
for idx in range(Nov):
    early_data=rx_data_bb[idx::Nov]
    center_data=rx_data_bb[idx+5::Nov]
    late_data=rx_data_bb[idx+10::Nov]
    early_data=early_data[0:len(late_data)]
    center_data=center_data[0:len(late_data)]
    slope_vector=(late_data - early_data)*np.sign(center_data)
    fcost.append(slope_vector.sum())

opt_time=Nov
for idx in range(len(fcost))[1:]:
    if np.sign(fcost[idx])<np.sign(fcost[idx-1]):
        opt_time=idx
        break
opt_time=opt_time

## R6 Sampling
bs_hat=np.sign(rx_data_diff[Nov+5+opt_time::Nov])
err = np.sum(bs-bs_hat[0:len(bs)])
print('BER=%.3f'%(err/Nbit))

## Post processing
# spectral density
#yswdn=signal.resample(ys,int(len(ys)/10))
#freq1, psd_rx = signal.welch(rx_data_pass,fs=10,nperseg=256,return_onesided=True)
freq2, psd_matched = signal.welch(rx_data_diff,fs=10,window='hann',nperseg=4096,return_onesided=True)

## PLoting
fig,ax=plt.subplots(4)
# ax[0].plot(ts[0:100],np.imag(tx_data_pass[0:100]))
# ax[0].plot(ts[0:100],np.real(tx_data_pass[0:100]))
ax[0].plot(t_prx[0:1100],Prx[0:1100])
ax[0].title.set_text('RSSI over 5 symbols')
# #ax[1].plot(ts,tx_data_ps)
#ax[1].plot(trx[0:108],np.sign(np.append(np.zeros(8),rx_data[0:100])))
#ax[1].plot(trx[0:108],np.sign(rx_data_diff[0:108]))
ax[1].plot(ts[0:100],tx_acc[0:100])
ax[1].plot(ts[0:100],rx_data[0:100])
ax[1].title.set_text('Accumulated Phase')
# ax[1].set_yticks(np.linspace(-3,3,7))
ax[1].grid()

for idx in range(rows):
    ax[2].plot(ts_eye,rx_data_eye[idx,:])

ax[2].title.set_text('Eye Diagram')
#ax[3].plot(freq1,20*np.log10(np.abs(psd_rx)))
ax[3].plot(freq2,20*np.log10(np.abs(psd_matched)))
ax[3].set_xlabel('freq')
ax[3].set_ylim([-120,50])
ax[3].title.set_text('Power Spectral Density')
fig.tight_layout()
plt.show()