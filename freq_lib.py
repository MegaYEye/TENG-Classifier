from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def show_fft(y, T_period):
    N = len(y)
    T = T_period
    x = np.linspace(0.0, N*T, N, endpoint=False)
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    plt.plot(xf, 20*np.log(2.0/N * np.abs(yf[0:N//2])))
    plt.grid()
    plt.show()


def apply_notch_filter1(xn, notch_freq=60, quality_factor=3, freq=489):
    notch_freq = notch_freq
    quality_factor = quality_factor
    samp_freq = freq
    b, a = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    ys = signal.filtfilt(b, a, xn)
    return ys

# Required input defintions are as follows;
# time:   Time between samples
# band:   The bandwidth around the centerline freqency that you wish to filter
# freq:   The centerline frequency to be filtered
# ripple: The maximum passband ripple that is allowed in db
# order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
#         IIR filters are best suited for high values of order.  This algorithm
#         is hard coded to FIR filters
# filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
# data:         the data to be filtered


def apply_notch_filter2(time, band, freq, ripple, order, filter_type, data):
    from scipy.signal import iirfilter, lfilter
    fs = 1/time
    nyq = fs/2.0
    low = freq - band/2.0
    high = freq + band/2.0
    low = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',
                     analog=False, ftype=filter_type)
    filtered_data = lfilter(b, a, data)
    return filtered_data