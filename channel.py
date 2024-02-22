'''
Junsang Yoo 2023.03.31
according to the paper:
"Frequency Offset Estimation and Correction
in the IEEE 802.11a WLAN "
'''
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()
wifi_preambles = loadmat('wifi_preambles.mat')

lstf = wifi_preambles['lstf'].flatten()     # short training field
lltf = wifi_preambles['lltf'].flatten()     # long training field
lsig = wifi_preambles['lsig'].flatten()     # signal field

## Basic Info
Fs = 20e6
Ts = 1/Fs
Fcarrier = 2.62e9
Ntotal = 1000
Npreamble = 400
Npilot = 64

ts = np.arange(Ntotal)*Ts
ts1 = np.arange(Npreamble)*Ts
ts2 = np.arange(128)*Ts
ts3 = np.arange(64)*Ts

## rx preamble
    # ref to amount of CFO
    # https://en.wikipedia.org/wiki/Carrier_frequency_offset
avg = []
#for i in range(1,51):
CFO = (np.random.rand()*80e-6-40e-6) * Fcarrier # -40ppm to +40ppm
#CFO = 0
#phs_off = np.random.rand()*2*np.pi-np.pi # -pi to +pi
# phs_off = 0
#channel_model = np.array([1, 0.1+0.3j]
SNR = 40


tf = np.hstack((lstf,lltf,lsig))
noisevar = tf.var()*10**(-SNR/20)/np.sqrt(2)

rx = np.zeros(Ntotal, dtype=np.complex128)
startidx = np.random.randint(300)
offset = np.exp(1j*(2*np.pi*CFO*ts1))
rx[startidx:startidx+Npreamble] = tf * offset
noise = np.random.randn(len(rx)*2).view(np.complex128)*noisevar
rx += noise

##channel fading
channel_model = np.array([1, 0.1+0.3j])
# channel_model = np.array([1])
rx = np.convolve(rx, channel_model)
H = np.fft.fft(channel_model, 64)


# Determine impulse response of channel
impulse_response = np.fft.fft(channel_model, n=2**12)
# plt.plot(np.fft.fftshift(abs(impulse_response)))
# plt.show()

##
pilot = lltf[:Npilot][::4] / lstf[:Npilot][::4]

##
# plot
# plt.subplot(211)
# plt.plot(rx.real)
# plt.subplot(212)
# plt.plot(np.abs(np.fft.fft(rx)))
# plt.show()

## Signal Detection
# RSSI
rxstartidx = 0
for idx in range(Ntotal):
    if np.abs(rx[idx]) > np.sqrt(noisevar) * 10:
        rxstartidx = idx
        break

## STF coarse correction
rxlstf = rx[rxstartidx:rxstartidx+160]
S1 = rxlstf[80:]
alphaST = 1/16/Ts*np.angle(np.sum(S1[:64].conj()*S1[16:])) # 앞뒤 64개씩 4chunk 16개의 샘플만큼(freq 일정하다고 가정)
print(f'carrier freq offset: {CFO} Hz')
print(f'coarse freq offset: {alphaST/2/np.pi} Hz')

## LTF fine correction
rxlltf = rx[rxstartidx+160:rxstartidx+320]
S2 = rxlltf[32:] / np.exp(1j*alphaST*ts2)
alphaLT = 1/64/Ts*np.angle(np.sum(S2[:64].conj()*S2[64:]))
#print(f'carrier freq offset: {CFO} Hz')
print(f'fine freq offset: {alphaLT/2/np.pi} Hz')
print(f'coarse + fine freq offset: {(alphaST+alphaLT)/2/np.pi} Hz')

avg.append(abs(CFO-(alphaST+alphaLT)/2/np.pi))
##
# # plt.stem(avg);
# # plt.show()

## Timing correction
lstf_xcorr = np.correlate(rxlstf[:16], lstf[:16], 'full')
rxstartidx += np.argmax(np.abs(lstf_xcorr))-15
end = time.time()
print(f"{end - start:.5f} sec")
# ## correction using pilot symbols
# rxlsig = rx[rxstartidx+320:rxstartidx+400]
#
# RXLSIG = np.fft.fft(rxlsig[16:]/np.exp(1j*(alphaST+alphaLT)*ts3))
#
# pilotidx = np.array([-21,-7,7,21])
# rxpilot = RXLSIG[pilotidx] * [1,1,1,-1]

## correction using pilot symbols
#rxlsig = rx[rxstartidx+320:rxstartidx+400]


## Channel Estimation and Correction using Pilot Symbols
# # rxdata = rx[rxstartidx + 320:] # start from end of LTF
# # data = rxdata[:Npilot*4] / lstf[Npilot*4::4] # remove CFO, downsample
# # data = rxdata[:Npilot*4] / lstf[:Npilot*4]
# #
# # pilot_matrix = data.reshape(-1, 4)[:,:Npilot//4] # split into 4 subcarriers, and use every 4th sample
# # H_est = np.mean(pilot_matrix, axis=0) / pilot # channel estimate
# # rxdata_corrected = np.zeros_like(rxdata)
# # for i in range(0, len(rxdata), 64):
# #     rxdata_corrected[i:i+64] = np.fft.ifft(np.fft.fft(rxdata[i:i+64]) / H_est)
# # rx_corrected = np.hstack((rx[:rxstartidx + 320], rxdata_corrected))
# #
# # ## Plotting
# # plt.subplot(311)
# # plt.plot(np.abs(np.fft.fft(rx)))
# # plt.subplot(312)
# # plt.plot(np.abs(np.fft.fft(rx_corrected)))
# # plt.subplot(313)
# # plt.plot(np.abs(np.fft.fft(H_est)))
# # plt.show()


# # rxlsig = lsig
#
# pilot_phase_error = np.angle(np.mean(rxpilot * np.conj(pilot)))
# alphaF = pilot_phase_error / (Npilot*Ts)
#
# print(f'coarse + fine + pilot freq offset: {(alphaST+alphaLT+alphaF)/2/np.pi} Hz')

##
# ltf_pilots = lltf[:Npilot][::4]
# rx_ltf_pilots = rx[rxstartidx+192:rxstartidx+192+128][::2]
# pilot_scale = np.mean(np.abs(ltf_pilots))/np.mean(np.abs(rx_ltf_pilots))
# rx_ltf_pilots_corrected = rx_ltf_pilots * np.exp(1j*np.angle(ltf_pilots))-np.mean(rx_ltf_pilots*np.exp(1j*np.angle(ltf_pilots)))
# alphaP = 1/np.pi*np.angle(np.sum(rx_ltf_pilots_corrected * np.conj(ltf_pilots)))
# print(f'pilot phase offset: {alphaP} rad')
#
# ## Apply correction
# rx_corrected = rx * np.exp(-1j*(alphaST+alphaLT+alphaP)*ts)
#
#
#
#
# RXLSIG = np.fft.fft(rxlsig[16:]/np.exp(1j*(alphaST+alphaLT)*ts3))
# #RXLSIG = np.fft.fft(rxlsig[:64])
# #RXLSIG = np.fft.fft(rxlsig[:64]/np.exp(2j*np.pi*CFO*ts3))
#

#
# pilot_values = []
# for idx in pilot_idx:
#     pilot_values.append(rxlsig[idx])
#
# alphaP = np.angle(pilot_values)    # phase offset
# alphaF = np.diff(alphaP) / (2*np.pi*pilot_interval*Ts)    # frequency offset
#
#
#
#
# plt.subplot(311)
# plt.plot(rx.real)
# plt.title('Received signal')
# plt.subplot(312)
# plt.plot(np.abs(np.fft.fft(rx)))
# plt.title('Spectrum of received signal')
# plt.subplot(313)
# plt.plot(np.abs(channel_response))
# plt.title('Estimated channel frequency response')
# plt.show()


## 나중에 다시 주석 풀기
# rxpilot_phs = np.unwrap(np.angle(rxpilot))  # assuming fine tuning is enough
# rx_phs_off = np.average(rxpilot_phs)
# # RXLSIG = RXLSIG*np.exp(-1j*rx_phs_off)
#
# plt.figure()
# plt.subplot(121)
# plt.plot(RXLSIG.real, RXLSIG.imag, 'o')
# plt.plot(RXLSIG[pilotidx].real, RXLSIG[pilotidx].imag, 'ro')
# plt.plot()
# plt.subplot(122)
# plt.stem(RXLSIG.real, label='real')
# plt.stem(RXLSIG.imag, label='imag')
# plt.stem(np.arange(64)[pilotidx], RXLSIG[pilotidx].real, label='pilot(real)', linefmt='r', markerfmt='ro')
# plt.stem(np.arange(64)[pilotidx], RXLSIG[pilotidx].imag, label='pilot(imag)', linefmt='g-.', markerfmt='g^')
# plt.legend()
# plt.show()