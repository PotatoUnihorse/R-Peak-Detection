import matplotlib.pyplot as plt
import numpy as np
import os
import math
import wfdb
import pywt
import scipy.signal


startSamp = 110000
endSamp = 120000
N = endSamp - startSamp

os.chdir('C:/Users/matth/PycharmProjects/pythonProject2/mit-bih-noise-stress-test-database-1.0.0')
sig, fields = wfdb.rdsamp('118e06', channels=[1], sampfrom=startSamp, sampto=endSamp)
sig2 = sig.flatten()
fs = fields.get('fs')

# next line figures out level of decomposition so that lowest approximation
# coefficient takes in frequencies that are ,<=0.5 Hz (usual cutoff for removing baseline wander)
level = math.floor(math.log(fs / 0.5, 2))
waveName = 'db6'
coeffs = pywt.wavedec(sig2, waveName, level=level)

# plots original signal
fig = plt.figure(figsize=(20, 15))
fig.add_subplot(level + 2, 1, 1)
plt.plot(sig)

# plots all coefficients below original signal
for i in range(level):
    fig.add_subplot(level + 2, 1, i + 2)
    plt.plot(coeffs[i])

# reconstructing the signal with noise removed
# cA4=pywt.downcoef('a',sig2,wavelet='db6',level=4)
cD1 = np.array(coeffs[level])
# THRESHOLD SELECTION
mediancD1 = np.median(cD1)
sigma = mediancD1 / .6457
t = sigma * math.sqrt(2 * math.log(N, 10))

# thresholds coefficients- need to explore this more
newCoeffs = [(pywt.threshold(subArr, t, mode='soft')) for subArr in coeffs]

# this section replaces cA8(lowest approx. coeff that contains baseline wander)
# and cD1(contains most of the high frequency noise) with arrays of zeros
cA = coeffs[1]
cD1 = coeffs[-1]
zerosA = np.zeros(cA.shape, dtype=float)
zerosD = np.zeros(cD1.shape, dtype=float)
newCoeffs[0] = zerosA
newCoeffs[-1] = zerosD

# reconstructs denoised signal without cA8 or cD1, with all other coeffs thresholded
denoised = pywt.waverec(newCoeffs, wavelet=waveName)

# plots original signal(top),denoised signal(middle), and clean signal(bottom)
fig2 = plt.figure(figsize=(20, 10))

fig2.add_subplot(4, 1, 1)
plt.plot(sig2)
plt.title('Original Signal')

fig2.add_subplot(4, 1, 2)
plt.plot(denoised)
plt.title('De-noised Signal')

fig2.add_subplot(4, 1, 3)

cleanSignal, field = wfdb.rdsamp('118', channels=[1], sampfrom=startSamp, sampto=endSamp)
cleanSig = cleanSignal.flatten()
plt.plot(cleanSig.flatten())
plt.title('Clean Signal')

# plt.show()

# maxima (list) -- contains all of the maximas in the clean signal // using the scipy.signal.find_peaks function
maxima = scipy.signal.find_peaks(cleanSig)
maxima = np.asarray(maxima)
maxima = maxima[0]
maxima = maxima.tolist()

# list that contains all of the minima in the clean signal // couldn't use scipy.signal.argrelmin because it wouldn't
# catch the minima where there is consecutive voltage readings
# read the clean signal data and find minima comparing neighboring values two spaces apart.
minima = list()
y = 0
for i in cleanSig[1:N-1]:

    if i < cleanSig[y-2] and i < cleanSig[y+2]:
        minima.append(y)

    y = y+1

# combine min and max list and sort
extrema = minima + maxima
extrema.sort()

# read list of all of the critical points
# loop through all values, finding the characteristic drop after the R peak. The 0.5 is arbitrary and should be changed?
# put all r_peak time values in probable peak list
L = len(extrema)
y = 0
probable_peaks = list()
for t in extrema[:L - 1]:
    t2 = extrema[y + 1]

    if cleanSig[t]-cleanSig[t2] > 0.5:  # need a threshold? value
        probable_peaks.append(t)
    y = y + 1


# after a QRS complex is detected, there is a 200 ms refractory period before the next one can be detected
# the remaining false positive peaks were the little peaks at point Q, before the R peak
# put all of the r_peak values in definitive peaks list
N = len(probable_peaks)
definitive_peaks = list()
y = 0
for g in probable_peaks[0:N-1]:
    g2 = probable_peaks[y + 1]

    if (g2 - g) > 150:  # 150 is an arbitrary number that worked for 118 and 119, should probably be changed
        definitive_peaks.append(g)
    y = y + 1

definitive_peaks.append(g2)  # the last peak was often left out, because of the loop and this was the only solution i
# could think of. shouldnt be a problem if were doing continuous monitoring, so i guess this can be commented out

fig2.add_subplot(4, 1, 4)
# display the plot of the clean signal, using red dots to identify the peaks
plt.plot(cleanSig)
plt.plot(definitive_peaks, cleanSig[definitive_peaks], 'ro')  # cleanSig --> ndarray (10000,)
# r_peaks --> list
plt.title('Detected R-peaks')

fig2.tight_layout(pad=4.0)
plt.show()
