import numpy as np

def read_DAC_data(filename):
    data = np.loadtxt(filename,skiprows=9)
    return data

