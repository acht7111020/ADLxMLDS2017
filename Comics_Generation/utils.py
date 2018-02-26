import numpy as np
import pickle
import random

class DataLoader():
    def __init__(self, path_x=None, path_y=None, batch_size=64, normal=True ):
        self.batch_size = batch_size
        self.step_x = pickle.load(open(path_x, 'rb'))
        self.step_y = pickle.load(open(path_y, 'rb'))
        if normal:
            self.step_y = np.asarray(self.step_y) 
        self.size = len(self.step_x)
        self.iters = np.ceil(len(self.step_x)/self.batch_size).astype(np.int32) 
        self.curr_index = 0
        print ('data size', self.size, ', data iters', self.iters)
        print(self.step_x.shape, self.step_y.shape)
        
    def next_batch(self):
        if self.curr_index + self.batch_size > self.size - 1:
            res_index = self.curr_index  + self.batch_size - self.size 
            batch_x = self.step_x[self.curr_index : self.size]  
            batch_y = self.step_y[self.curr_index : self.size]  
            self.curr_index = 0
        else:
            batch_x = self.step_x[self.curr_index : self.curr_index + self.batch_size]  
            batch_y = self.step_y[self.curr_index : self.curr_index + self.batch_size]  
            self.curr_index += self.batch_size
        x_wrong = self.step_x[random.sample(range(self.size), len(batch_x))]
        y_wrong = self.step_y[random.sample(range(self.size), len(batch_x))]
        return np.asarray(batch_x).astype(np.float32), np.asarray(x_wrong).astype(np.float32), np.asarray(batch_y).astype(np.float32), np.asarray(y_wrong).astype(np.float32)

    def first_batch(self):
        batch_x = self.step_x[0 : self.batch_size]  
        batch_y = self.step_y[0 : self.batch_size]  
        return np.asarray(batch_x).astype(np.float32), np.asarray(batch_y).astype(np.float32)

    def random_sample(self): # return 1450*4 samples
        rand_cap = random.sample(range(self.size), self.batch_size)
        batch_x = self.step_x[rand_cap]
        batch_y = self.step_y[rand_cap]
        x_wrong = self.step_x[random.sample(range(self.size), self.batch_size)]
        y_wrong = self.step_y[random.sample(range(self.size), self.batch_size)]

        return np.asarray(batch_x).astype(np.float32), np.asarray(x_wrong).astype(np.float32), np.asarray(batch_y).astype(np.float32), np.asarray(y_wrong).astype(np.float32)
