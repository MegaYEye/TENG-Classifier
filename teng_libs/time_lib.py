import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths, butter,filtfilt
import matplotlib.pyplot as plt



def lpf(x):
    b, a = butter(4, 0.05, 'low', fs=1)
    x = filtfilt(b,a,x)
    return x


def nms(x_abs,peaks, half_window_size):
    if len(peaks)==0:
        return np.array([])
    masks = []
    eps = np.finfo(np.float64).eps
    for k in peaks:
        left = k - half_window_size
        right = k + half_window_size
        if left<0:
            left = 0
        if right>=len(x_abs):
            right = len(x_abs)-1
            
        peaks_local = peaks[np.logical_and(peaks>=left, peaks<right)]
        segment_max = x_abs[peaks_local].max()
        if x_abs[k]<segment_max-eps:
            masks.append(False)
        else:
            masks.append(True)
    peaks_a = peaks[np.array(masks)]
    # next: remove equal points
    peaks_b = []
    last = -np.inf
    for k in peaks_a:
        if k-last<half_window_size:
            continue
        last = k
        peaks_b.append(k)
    return np.array(peaks_b)

def width_filter1(x_abs, peaks, min_height, half_width):
    if len(peaks)==0:
        return np.array([])
    masks=[]
    for p in peaks:
        left = p-half_width
        right = p+half_width
        right2 = p+half_width*2
        left2 = p-half_width*2
        if left<0:
            left = 0
        if left2<0:
            left2=0
        if right>=len(x_abs):
            right = len(x_abs)-1
        if right2>=len(x_abs):
            right2 = len(x_abs)-1
        mask = (x_abs[left:right]>min_height).all()
        if not mask:
            mask = (x_abs[p:right2]>min_height).all()
        if not mask:
            mask = (x_abs[left2:p]>min_height).all()
        masks.append(mask)
    masks = np.array(masks)
    return peaks[masks]

def width_filter2(x_abs, peaks, min_width):
    if len(peaks)==0:
        return np.array([])
    w,_,_,_ = peak_widths(x_abs, peaks,rel_height=1.0)
    return peaks[w>min_width]


def get_peaks(x, height=1e-10, width=10, nms_pts=700):
    x_abs = np.abs(x)
    peaks, _ = find_peaks(x_abs, height=height,prominence=height)
    peaks = width_filter1(x_abs,peaks, min_height=1e-10, half_width=width//2)
    peaks = width_filter2(x_abs, peaks, width)
    peaks = nms(x_abs,peaks, nms_pts)
    peaks.sort()
    return peaks

def segment(x, peak, left, right):
    l = peak-left
    l[l<0]=0
    r = peak+right
    r[r>len(x)-1]=len(x)-1
    result =  np.array([x[ll:rr] for ll,rr in zip(l,r)])
    return result

    
if __name__ == '__main__':
    from DAC_lib import  read_DAC_data
    #! tape
    filename = "data_DAC_structure/tape/hexagonal wrench with double side tape.data"
    # filename = "data_DAC_structure/tape/screw head with double side tape.data"
    # filename = "data_DAC_structure/tape/tape edge.data"
    # filename = "data_DAC_structure/tape/tape side.data"

    #! foam
    # filename = "data_DAC_structure/foam/foam edge.data"
    # filename = "data_DAC_structure/foam/foam side.data"
    data = read_DAC_data(filename)
    x = data[:,1]
    # x = lpf(x)
    peaks = get_peaks(x)
    peaks = peaks[::2]
    plt.plot(np.arange(len(x)), x)
    plt.plot(peaks, x[peaks], "x")
    plt.show()