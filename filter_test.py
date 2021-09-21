import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from scipy import signal

from scipy.fft import fft, fftfreq
from pyts.classification import BOSSVS
from pyts.classification import SAXVSM
# from pyts.classification import LearningShapelets


def show_fft(y):
    N = len(y)
    T = 1/489
    x = np.linspace(0.0, N*T, N, endpoint=False)
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    plt.plot(xf, 20*np.log(2.0/N * np.abs(yf[0:N//2])))
    plt.grid()
    plt.xlabel("HZ")
    plt.ylabel("magnitude")
    plt.show()


def apply_filter(xn):
    notch_freq = 60
    quality_factor = 3
    samp_freq = 489
    b, a = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    # zi = signal.lfilter_zi(b, a) * xn[0]
    # ys = []
    # for x in xn:
    # y, zi = signal.lfilter(b, a, np.array([x]), zi=zi)
    # ys.append(y)
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


def Implement_Notch_Filter(time, band, freq, ripple, order, filter_type, data):
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


def generate_data(folder):
    """
    format of the folder: with files, with class_name.csv
    """
    files = glob(f"{folder}/*.csv")
    bufs = []
    bufs_np = []
    ys = []
    for f in files:
        id_name = os.path.basename(f)
        idx, fullname = id_name.split("_")
        classname = fullname.split(".")[0]
        csv = pd.read_csv(f, header=None)
        csv.columns = [f"ch{x}" for x in range(9)]
        print(idx, classname)
        X1 = csv["ch0"].values
        X2 = csv["ch1"].values
        # X_fil = apply_filter(X1)
        X_fil = Implement_Notch_Filter(1/489, 5,60,10,6,'butter', X1)
        X_fil = Implement_Notch_Filter(1/489, 40,180,10,6,'butter', X_fil)
        # sos = signal.iirfilter(17, [50, 70], rs=60, btype='bandstop',
        #                 analog=False, ftype='cheby2', fs=489,
        #                 output='sos')
        # X_fil = signal.sosfilt(sos, X1)
        show_fft(X1)
        show_fft(X_fil)
 
        plt.plot(np.arange(len(X1)),X1)
        plt.plot(np.arange(len(X_fil)),X_fil)
        plt.show()


if __name__ == '__main__':
    folder = "./data"
    generate_data(folder)
