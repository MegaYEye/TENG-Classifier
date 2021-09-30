import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from scipy import signal

# from pyts.classification import LearningShapelets

from sktime.classification.interval_based import TimeSeriesForestClassifier
# from sktime.datasets import load_arrow_head,load_japanese_vowels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sktime.utils.data_processing import from_2d_array_to_nested, from_3d_numpy_to_nested
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.shapelet_based import MrSEQLClassifier
from sklearn.pipeline import Pipeline

# from sktime.datasets import load_basic_motions
from sktime.transformations.panel.compose import ColumnConcatenator


def show_fft(y):
    N = len(y)
    T = 1/489
    x = np.linspace(0.0, N*T, N, endpoint=False)
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    plt.plot(xf, 20*np.log(2.0/N * np.abs(yf[0:N//2])))
    plt.grid()
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
        # X_fil = apply_filter(X)
        # X_fil = Implement_Notch_Filter(1/489, 20,60,110,6,'butter', X)
        # show_fft(X)
        # show_fft(X_fil)
        # print()
        X1 = X1 - np.mean(X1)
        X2 = X2 - np.mean(X2)
        X1abs = abs(X1)
        X2abs = abs(X2)
        # plt.plot(np.arange(len(X1)),X1)
        # plt.plot(np.arange(len(X2)),X2)
        # plt.show()

        i = 100
        while i < len(X1):
            if (X1abs[i]+X2abs[i]) > 15:
                X1bar = X1[i-50:i+300]
                X2bar = X2[i-50:i+300]
                bufs.append({"name": classname, "X1": X1bar, "X2": X2bar})
                bufs_np.append([X1bar, X2bar])
                ys.append(int(idx)-1)
                i += 300
                # plt.figure()
                # plt.plot(np.arange(len(X1bar)), X1bar)
                # plt.plot(np.arange(len(X2bar)), X2bar)
                # plt.show()
                # plt.close("all")
            else:
                i += 1
    return np.array(bufs_np), np.array(ys)

    # plt.plot(np.arange(len(X_fil)), X_fil)
    # plt.show()


if __name__ == '__main__':
    folder = "./data"
    X, y = generate_data(folder)
    # y = y[:,None]
    # X = from_3d_numpy_to_nested(X)
    # y = from_2d_array_to_nested(y)

    # classifier(data)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5,random_state=123)
    mask_train = [np.argwhere(y==0)[0:1].flatten(), np.argwhere(y==1)[0:7].flatten(), np.argwhere(y==2)[0:7].flatten()]
    mask_train = np.hstack(mask_train).flatten()
    mask_test = np.setdiff1d(np.arange(len(y)), mask_train)
    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]
    
    X_train = from_3d_numpy_to_nested(X_train)
    X_test = from_3d_numpy_to_nested(X_test)
 
    
    from sktime.classification.shapelet_based import MrSEQLClassifier
    clf = MrSEQLClassifier()
    clf.fit(X_train, y_train)
    print("MrSEQLClassifier")
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    print(y_test)
    print(y_pred)

    # clf = ColumnEnsembleClassifier(
    #     estimators=[
    #         ("TSF0", TimeSeriesForestClassifier(n_estimators=100), [0]),
    #         ("BOSSEnsemble3", BOSSEnsemble(max_ensemble_size=5), [3]),
    #     ]
    # )
    # clf.fit(X_train, y_train)
    # print("ColumnEnsembleClassifier")
    # print(clf.score(X_test, y_test))

    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", TimeSeriesForestClassifier(n_estimators=100)),
    ]
    clf = Pipeline(steps)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    print("Concatenate")
    print(clf.score(X_test, y_test))
