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
        plt.plot(np.arange(len(X1)),X1)
        plt.plot(np.arange(len(X2)),X2)
        plt.show()

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


def classification_BOSSVS(train_X, train_y, test_X, test_y):
    # clf = LearningShapelets(random_state=42, tol=0.01)
    X_train = train_X[:, 1, :] - train_X[:, 0, :]
    X_test = test_X[:, 1, :] - test_X[:, 0, :]
    y_train = train_y
    y_test = test_y
    # clf.fit(train_X, train_y)
    # print(clf.test(test_X))
    # print(test_y)
    bossvs = BOSSVS(word_size=2, n_bins=3, window_size=10)
    bossvs.fit(X_train, y_train)
    # tfidf = bossvs.tfidf_
    # vocabulary_length = len(bossvs.vocabulary_)
    # X_new = bossvs.decision_function(X_test)
    # print(X_new)

    # clf = LearningShapelets(random_state=42, tol=0.01)
    # clf.fit(X_train, y_train)
    # print(clf.score(X_test,y_test))
    # print(clf.score(X_train,y_train))
    print("train accuracy:",bossvs.score(X_train,y_train))
    print("test accuracy:",bossvs.score(X_test,y_test))
    print("predict y:",bossvs.predict(X_test))
    print("true y:",y_test)
    
    
# def classification_SAXVSM(train_X, train_y, test_X, test_y):
#     # clf = LearningShapelets(random_state=42, tol=0.01)
#     X_train = train_X[:, 1, :] - train_X[:, 0, :]
#     X_test = test_X[:, 1, :] - test_X[:, 0, :]
#     y_train = train_y
#     y_test = test_y
#     # clf.fit(train_X, train_y)
#     # print(clf.test(test_X))
#     # print(test_y)
#     saxvsm = SAXVSM(window_size=15, word_size=3, n_bins=2,
#                     strategy='uniform')
#     saxvsm.fit(X_train, y_train)
#     # tfidf = bossvs.tfidf_
#     # vocabulary_length = len(bossvs.vocabulary_)
#     # X_new = bossvs.decision_function(X_test)
#     # print(X_new)

#     # clf = LearningShapelets(random_state=42, tol=0.01)
#     # clf.fit(X_train, y_train)
#     # print(clf.score(X_test,y_test))
#     # print(clf.score(X_train,y_train))
#     print(saxvsm.score(X_test,y_test))
#     print(saxvsm.score(X_train,y_train))
#     print(saxvsm.predict(X_test))
#     print(y_test)
    
if __name__ == '__main__':
    folder = "./data"
    X, y = generate_data(folder)
    train_select = np.arange(0, len(X), 2)
    test_select = np.setdiff1d(np.arange(len(X)), train_select)
    train_X, test_X = X[train_select], X[test_select]
    train_y, test_y = y[train_select], y[test_select]
    print(X.shape, y.shape)
    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)
    classification_BOSSVS(train_X, train_y, test_X, test_y)

    # classifier(data)
