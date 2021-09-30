from teng_libs.time_lib import get_peaks, segment, lpf
from teng_libs.DAC_lib import read_DAC_data
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
datafiles = [
    # "data_DAC_structure/tape/hexagonal wrench with double side tape.data",
    # "data_DAC_structure/tape/screw head with double side tape.data",
    "data_DAC_structure/tape/tape edge.data",
    # "data_DAC_structure/tape/tape side.data",
]

labels = [
    "wrench", 
    # "screw", 
    # "tape_edge", 
    # "tape_side"
    ]
colors=[
    "r",
    # "g",
    # "b",
    # "y"
    ]
# datafiles = [
#     # "data_DAC_structure/foam/foam edge.data",
#     "data_DAC_structure/foam/foam side.data"
# ]
# labels=["edge", "side"]
# colors=["r","g"]

# segments = []
plt.figure()
for f,lb,c in zip(datafiles, labels, colors):
    x = read_DAC_data(f)
    x = x[:,1]
    peaks = get_peaks(lpf(x))
    peaks_contact = peaks[::2]
    ss = segment(x, peaks_contact,300,500)
    for s in ss:
        plt.plot(np.arange(len(s))*0.001,s)
    # segments.append(ss)
# plt.legend()
plt.show()    



    