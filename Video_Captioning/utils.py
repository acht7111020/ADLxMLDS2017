# from joblib import Parallel, delayed
import _pickle as pickle
import json
import time
import numpy as np
import random
from random import shuffle

class DataLoader():
    def __init__(self, x, y=None, batch_size=128, fsize=80, emb_dim=300, test=False):
        start = time.time()
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.fsize = fsize
        
        word_dict = pickle.load(open('data/word_dict.p', 'rb'))
        self.itow = word_dict['itow']
        self.wtoi = word_dict['wtoi']
        self.vocabsize = len(self.itow) + 1 
        print('vocabsize:', self.vocabsize)

        self.init_pretrain_wordemb()

        if test:
            self.step_x = x 
            self.size = len(self.step_x)
            self.iters = np.ceil(len(self.step_x)/self.batch_size).astype(np.int32)
            end = time.time()
            print('DataLoader use time %4.1fs' % (end-start))
            print('step x size:', len(self.step_x), 'shape', self.step_x[0].shape, 'iter', self.iters)   
            # print('step y size:', len(self.step_y), 'shape', self.step_y[0].shape, 'caption_len', self.caption_len)
            self.curr_index = 0
        else:

            self.caption_len = 41
            self.x = x # dict: {id: features (80*4096)}
            self.y = y # dict: {id: caption list}
            
            end = time.time()
            print('DataLoader use time %4.1fs' % (end-start))

    def init_pretrain_wordemb(self):
        w2v_path = 'pretrain/glove.6B.' + str(self.emb_dim) + 'd.txt'
        embeddings_index = {}
        f = open(w2v_path, 'r', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        # prepare embedding matrix
        embedding_matrix = np.random.rand(self.vocabsize, self.emb_dim)
        for word, idx in self.wtoi.items():
            if word in embeddings_index.keys():
                embedding_matrix[idx] = embeddings_index[word]
                
        self.pretrain_wordemb = embedding_matrix
 
    def random_sample(self, sample_num, cutting_size=45): # return 1450*4 samples
        batch_y = []
        batch_x = []
        for idx, features in self.x.items():
            
            video_y = self.y[idx]

            rand_cap = random.sample(range(len(video_y)), sample_num)
            for i in range(0, sample_num):
                if len(video_y[rand_cap[i]]) < cutting_size: # train shorter than longer
                    batch_x.append(features)
                    batch_y.append(video_y[rand_cap[i]])

        y_seq_len = get_sequence_lengths(batch_y)
        y = make_sequences_same_length(batch_y, y_seq_len) 
        c = list(zip(batch_x, y))
        shuffle(c)
        self.step_x, self.step_y = zip(*c)
 
        self.size = len(self.step_x)
        self.iters = np.ceil(len(self.step_x)/self.batch_size).astype(np.int32)
        self.curr_index = 0

    def next_batch(self):
        #for i in range(0, self.iters):
        if self.curr_index  + self.batch_size > self.size  - 1:
            res_index = self.curr_index  + self.batch_size - self.size 
            batch_x = np.concatenate((self.step_x [self.curr_index  : self.size ], np.zeros([res_index, self.fsize, 4096])), axis = 0) #self.step_x [0: res_index]), axis = 0)
            batch_y = np.concatenate((self.step_y [self.curr_index  : self.size ], np.zeros([res_index, self.caption_len])), axis = 0) #self.step_y [0: res_index]), axis = 0)
            self.curr_index  = 0
        else:
            batch_x = self.step_x [self.curr_index  : self.curr_index  + self.batch_size]  
            batch_y = self.step_y [self.curr_index  : self.curr_index  + self.batch_size]  
            self.curr_index  += self.batch_size

        return np.asarray(batch_x).astype(np.float32), np.asarray(batch_y).astype(np.int32)

    def next_batch_test(self):
        if self.curr_index  + self.batch_size > self.size  - 1:
            res_index = self.curr_index  + self.batch_size - self.size 
            batch_x = np.concatenate((self.step_x [self.curr_index  : self.size ], self.step_x [0: res_index]), axis = 0)
            self.curr_index  = 0
        else:
            batch_x = self.step_x [self.curr_index  : self.curr_index  + self.batch_size]  
            self.curr_index  += self.batch_size

        return np.asarray(batch_x).astype(np.float32)

def make_sequences_same_length(sequences, sequences_lengths, default_value=0.0, max_length=41):
    """
    Make sequences same length for avoiding value
    error: setting an array element with a sequence.
    Args:
        sequences: list of sequence arrays.
        sequences_lengths: list of int.
        default_value: float32.
            Default value of newly created array.
    Returns:
        result: array of with same dimensions [num_samples, max_length, num_features].
    """

    # Get number of sequnces.
    num_samples = len(sequences)

    if max_length == 0:
        max_length = np.max(sequences_lengths)

    # Get shape of the first non-zero length sequence.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    # Create same sizes array
    result = (np.ones((num_samples, max_length) + sample_shape) * default_value)

    # Put sequences into new array.
    for idx, sequence in enumerate(sequences):
        result[idx, :len(sequence)] = sequence

    return result

def get_sequence_lengths(inputs):
    """
    Get sequence length of each sequence.
    Args:
        inputs: list of lists where each element is a sequence.
    Returns:
        array of sequence lengths.
    """
    result = []
    for input in inputs:
        result.append(len(input))

    return np.array(result, dtype=np.int64)

