from teng_libs.time_lib import get_peaks, segment, lpf, get_all_peaks
from teng_libs.DAC_lib import read_CSV_data
import numpy as np
import matplotlib.pyplot as plt
datafiles = [
    ["data3/tape1.csv",
    "data3/tape2.csv"],
    ["data3/thorlab-bulk1.csv",
     "data3/thorlab-bulk2.csv",
     "data3/thorlab-bulk3.csv"],
]

labels = [
    "tape", 
    "thorlab", 
    ]



for flist,lb in zip(datafiles, labels):
    fconcat = []
    for f in flist:
        data = read_CSV_data(f)
        fconcat.append(data)
    x = np.hstack(fconcat)
    peaks = get_all_peaks(x)
    
    plt.figure()
    peaks_contact = peaks[::2]
    ss = segment(x, peaks_contact,left=50,right=100)
    for s in ss:
        plt.plot(np.arange(len(s))*0.001,s)
    plt.title(lb+"_attach")
    plt.show()    
    
    plt.figure()
    peaks_contact = peaks[1::2]
    ss = segment(x, peaks_contact,left=50,right=100)
    for s in ss:
        plt.plot(np.arange(len(s))*0.001,s)
    plt.title(lb+"_detach")
    plt.show()    





    