import time
import numpy as np

# timer class for timing projecting time-to-finish
class Timer(object):
    def __init__(self,name=None, verbosity=1):
        self.name = name
        self.finished = 0
        self.times = []
        self.verbosity = verbosity
    
    def __enter__(self):
        self.tstart = time.time()
    
    def __exit__(self, type, value, traceback):
        self.elapsed = time.time() - self.tstart
        self.times.append(self.elapsed)
        if self.verbosity == 1:
           self.print_times()
    
    def get_times(self):
        return self.times

    def print_times(self):
        print('--- {} -> Segment Time: {:5.2f}s - Total Time: {:5.2f}s ---'.format(self.name, self.elapsed, np.sum(self.times)))
    
    def set_verbosity(self, verbosity):
        self.verbosity = verbosity