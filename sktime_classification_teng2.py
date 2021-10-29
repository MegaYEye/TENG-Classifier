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
from teng_libs.time_lib import get_peaks, segment, lpf, get_all_peaks
from teng_libs.DAC_lib import read_CSV_data



def generate_data(folder):
    """
    format of the folder: with files, with class_name.csv
    """
    datafiles = [
        ["data3/tape1.csv",
        "data3/tape2.csv",],
        ["data3/thorlab-bulk1.csv",
        "data3/thorlab-bulk2.csv",
        "data3/thorlab-bulk3.csv",]
    ]

    labels = [
        0,
        1 
    ]
    xs = []
    ys = []
    for flist,lb in zip(datafiles, labels):
        fconcat = []
        for f in flist:
            data = read_CSV_data(f)
            fconcat.append(data)
        x = np.hstack(fconcat)
        peaks = get_all_peaks(x)
        
        plt.figure()
        #! for attach
        peaks_contact = peaks[::2]
        #! for detach
        # peaks_contact = peaks[1::2]
        ss = segment(x, peaks_contact,left=50,right=100)
        xs.append(ss)
        ys.extend([lb]* len(ss))
    return np.vstack(xs), np.array(ys)


if __name__ == '__main__':
    folder = "./data3"
    X, y = generate_data(folder)
    # y = y[:,None]
    # X = from_3d_numpy_to_nested(X)
    # y = from_2d_array_to_nested(y)

    # classifier(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10,random_state=123)
    # mask_train = [np.argwhere(y==0)[0:1].flatten(), np.argwhere(y==1)[0:7].flatten(), np.argwhere(y==2)[0:7].flatten()]
    # mask_train = np.hstack(mask_train).flatten()
    # mask_test = np.setdiff1d(np.arange(len(y)), mask_train)
    # X_train, y_train = X[mask_train], y[mask_train]
    # X_test, y_test = X[mask_test], y[mask_test]
    
    X_train = from_3d_numpy_to_nested(X_train[:,None,:])
    X_test = from_3d_numpy_to_nested(X_test[:,None,:])
 
    
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
