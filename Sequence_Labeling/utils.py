from __future__ import print_function, division
import numpy as np

class DataLoader():
    def __init__(self, x, y=None, batch_size=128, timestep=5, num_class=47, build=True):
        self.timestep = timestep
        self.batch_size = batch_size
        self.num_class = num_class

        # for sequence training 
        self.size = len(x)
        self.iters = np.ceil(len(x)/self.batch_size).astype(np.int32)
        self.step_x = x
        self.step_y = y
        print ('data size', self.size, ', data iters', self.iters)
        self.curr_index = 0
        
    def next_batch_seq(self):
        if self.curr_index + self.batch_size > self.size - 1:
            batch_x = self.step_x[self.curr_index : self.size]
            batch_y = self.step_y[self.curr_index : self.size]
            self.curr_index = 0
        else:
            batch_x = self.step_x[self.curr_index : self.curr_index + self.batch_size]  
            batch_y = self.step_y[self.curr_index : self.curr_index + self.batch_size]  
            self.curr_index += self.batch_size 
            
        return batch_x, batch_y

    def next_batch_test_seq(self):
        if self.curr_index + self.batch_size > self.size - 1:
            batch_x = self.step_x[self.curr_index : self.size]
            self.curr_index = 0
        else:
            batch_x = self.step_x[self.curr_index : self.curr_index + self.batch_size]  
            self.curr_index += self.batch_size

        return batch_x

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

def sparse_tuples_from_sequences(sequences, dtype=np.int32):
    """
    Create a sparse representations of inputs.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indexes = []
    values = []

    for n, sequence in enumerate(sequences):
        indexes.extend(zip([n] * len(sequence), range(len(sequence))))
        values.extend(sequence)

    indexes = np.asarray(indexes, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indexes).max(0)[1] + 1], dtype=np.int64)

    return indexes, values, shape

def make_sequences_same_length(sequences, sequences_lengths, default_value=0.0):
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
