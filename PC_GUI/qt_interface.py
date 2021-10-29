import sys
# from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib import animation
import random
import numpy as np
import threading
from teng_driver import sensor_threadfunc
from copy import deepcopy
n_channel = 4
lock = threading.Lock()
class Window(QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.counter = 0
        self.n_channel = n_channel
        # a figure instance to plot on
        self.figure = fig = plt.figure()
        
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0,3000)
        self.ax.set_ylim(0,1000)
        
        self.plot_lines = []
        for i in range(n_channel):
            plot_line, = self.ax.plot([], [], label = f"ch_{i}",linewidth=0.3)
            self.plot_lines.append(plot_line)
            plot_line.set_data([], [])
        self.ax.legend(loc=2, prop={'size': 6})
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `plot` method
        self.button = QPushButton('Connect')
        self.button2 = QPushButton('Robot Sample')
        self.button.clicked.connect(self.connect)
        self.button2.clicked.connect(self.sample)

        # set the layout
        self.wordLabel = QLabel(self)
        self.wordLabel.setStyleSheet('font-size: 18pt; color: blue;')
        self.wordLabel.setGeometry(20, 20, 100, 50)
        
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        layout.addWidget(self.button2)
        layout.addWidget(self.wordLabel)
        self.setLayout(layout)

        timer = QTimer(self)
        self.timer = timer
        timer.timeout.connect(self.onTimeout)
        timer.start(1000//30)
        
        self.thread_param = {"run":True}
        self.sensor_output = [[] for x in range(n_channel)]
        self.t = threading.Thread(target=sensor_threadfunc, args=(self.thread_param, self.sensor_output))
        self.t.start()
        
        
    def onTimeout(self):
        self.wordLabel.setText("placeholder: recognition result")
        self.counter += 1
        min_axis = np.inf
        max_axis = -np.inf
        with lock:
            for i in range(n_channel):
                if self.sensor_output[i]:
                    if len(self.sensor_output[i])>3000:
                        self.sensor_output[i] = self.sensor_output[i][-3000:]
                        
        for i in range(n_channel):
            sout = deepcopy(self.sensor_output[i])
            if np.min(sout)-10<min_axis:
                min_axis = np.min(sout)-10
            if np.max(sout)+10>max_axis:
                max_axis = np.max(sout)+10
            self.plot_lines[i].set_data(np.arange(len(sout)), sout)
        self.ax.set_ylim(min_axis, max_axis)
        self.figure.canvas.draw()
        

        

    def connect(self):
        ''' plot some random stuff '''
        for i in range(n_channel):
            y = [random.random() for i in range(10)]
            x = np.arange(len(y))
            self.plot_lines[i].set_data(x, y)
            
    def sample(self):
        pass

 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())