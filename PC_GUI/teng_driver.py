import serial
import time
import numpy as np
from copy import deepcopy
import threading

lock = threading.Lock()
n_channel=4

def configuration():
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = '/dev/ttyACM0'
    ser.close()
    ser.open()
    ser.flush()
    return ser

class Sensor(object):        
    def __init__(self, n_channel=4):
        self.ser = configuration()
        self.n_channel=n_channel
        self.sensor_data = [[] for _ in range(n_channel)]
        self.update_time = time.time()
        self.number = 0
        self.buf = ""
    
    def close(self):
        self.ser.close()
        
    def get_data(self):
        s_data = np.array(self.sensor_data)
        self.sensor_data = [[] for _ in range(self.n_channel)]
        return s_data
                
    def read(self):
        bytes1 = self.ser.read(257).decode()
        self.buf += bytes1
        
    def find_start_end(self):
        start = -1
        end = -1
        for i in range(len(self.buf)):
            if self.buf[i] == "$":
                start = i
                for j in range(len(self.buf[i::])):
                    if self.buf[i+j] == ";":
                        end = i+j
                        return start, end        
        return start, end
    
    def pop_reading(self):
        s = self.sensor_data
        self.sensor_data = [[] for _ in range(self.n_channel)]
        return s
    
    def parse(self):
        while True:
            start, end = self.find_start_end()
            if start>=0 and end>=0:
                substr = self.buf[start:end]
                self.buf = self.buf[end::]
                substr = substr.replace("$"," ")
                data = substr.split(" ")
                # print(data)
                for k in range(len(data[1::])):
                    self.sensor_data[k].append(int(data[1+k]))
            else:
                break


def sensor_threadfunc(param, output_state):
    s = Sensor(n_channel=n_channel)
    while param["run"]:
        time.sleep(0.01)
        s.read()
        s.parse()
        with lock:
            ss = s.pop_reading()
            for x in range(n_channel):
                output_state[x].extend(ss[x])
    s.close()
        
    

if __name__ == '__main__':
    param = {"run":True}
    output_state = [[] for x in range(n_channel)]
    t = threading.Thread(target=sensor_threadfunc, args=(param, output_state))
    t.start()
    while True:
        print(len((output_state[0])))
        time.sleep(0.01)
        