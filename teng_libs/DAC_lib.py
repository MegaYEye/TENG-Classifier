import numpy as np

def read_DAC_data(filename):
    data = np.loadtxt(filename,skiprows=9)
    return data

def read_CSV_data(filename, usecols=range(1)):
    data = np.loadtxt(filename,delimiter=",",usecols=usecols)
    return data

if __name__ == '__main__':
    data = read_CSV_data("/media/ye/1494EA6E94EA5232/programming2/TimeSeqTENG/data2/cardbox.csv")
    print(data.shape)
    
