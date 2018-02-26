import numpy as np
import tensorflow as tf

rnn = tf.contrib.rnn
slim = tf.contrib.slim
from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops

def swish_sigmoid(x, name=None):
    x = ops.convert_to_tensor(x, name="x")
    return x*gen_math_ops._sigmoid(x, name=name)

class RNN_LSTM():
    def __init__(self, batch_size, timestep, num_classes, num_input, learning_rate, num_hidden, trainable=True):
        self.lr = learning_rate
        self.bs = batch_size
        self.n_input = num_input
        self.n_hidden = num_hidden
        self.n_classes = num_classes
        self.timestep = timestep
        self.name = 'RNN'
        self.trainable = trainable

        self.x_train = tf.placeholder("float", [None, None, self.n_input]) # (batchsize, timestep, n_input) 
        self.y_train = tf.placeholder(tf.int32, [None, None, self.n_classes])# (batchsize, timestep, n_class)  
        self.seq_len = tf.placeholder(tf.int32, [None])  
        self.n_hidden = 512
    def build(self): 

        self.logits = self.RNN(self.x_train) #(batch_size * timestep, n_classes)
        self.so_max = tf.nn.softmax(self.logits)    
        self.pred = tf.argmax(self.logits, 1) # (batch * timestep, 1) 

        self.y = tf.reshape(self.y_train, [-1, self.n_classes])
        # softmax_cross_entropy_with_logits ctc_loss
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, 
            labels=self.y))

        self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr, epsilon=1e-08, decay=0.0).minimize(self.loss_op) #, epsilon=1e-08, decay=0.0


    def RNN(self, x):

        # bi-directional
        lstm_fw_cell1 = rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True, initializer=xavier_initializer(uniform=True))
        lstm_bw_cell1 = rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True, initializer=xavier_initializer(uniform=True))
        lstm_fw_cell2 = rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True, initializer=xavier_initializer(uniform=True))
        lstm_bw_cell2 = rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True, initializer=xavier_initializer(uniform=True))
        lstm_fw_cell3 = rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True, initializer=xavier_initializer(uniform=True))
        lstm_bw_cell3 = rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True, initializer=xavier_initializer(uniform=True))
        lstm_fw_cell4 = rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True, initializer=xavier_initializer(uniform=True))
        lstm_bw_cell4 = rnn.LSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True, initializer=xavier_initializer(uniform=True)) 

        output, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell1, lstm_bw_cell1, 
            x, sequence_length=self.seq_len, dtype=tf.float32, scope='bidir1') 

        output = tf.concat(output, 2)
        output = slim.fully_connected(output, self.n_hidden, activation_fn=None)

        output, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell2, lstm_bw_cell2, 
            output, initial_state_fw=states[0], initial_state_bw=states[1], 
            sequence_length=self.seq_len, dtype=tf.float32, scope='bidir2') 

        output = tf.concat(output, 2)
        if self.trainable:
            output = slim.dropout(output, 0.5, scope='dropout1')
        output = slim.fully_connected(output, self.n_hidden, activation_fn=None)

        output, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell3, lstm_bw_cell3, 
            output, initial_state_fw=states[0], initial_state_bw=states[1], 
            sequence_length=self.seq_len, dtype=tf.float32, scope='bidir3') 

        output = tf.concat(output, 2)
        # output = slim.dropout(output, 0.7, scope='dropout2')
        output = slim.fully_connected(output, self.n_hidden, activation_fn=None)

        output, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell4, lstm_bw_cell4, 
            output, initial_state_fw=states[0], initial_state_bw=states[1], 
            sequence_length=self.seq_len, dtype=tf.float32, scope='bidir4') 

        output = tf.concat(output, 2)
        if self.trainable:
            output = slim.dropout(output, 0.5, scope='dropout2') 
        output = tf.reshape(output, [-1, self.n_hidden*2])

        return slim.fully_connected(output, self.n_classes, activation_fn=None) # x output: (batch*timestep, n_hidden)

class CNN_RNN():
    def __init__(self, batch_size, timestep, num_classes, num_input, learning_rate, num_hidden):
        self.lr = learning_rate
        self.bs = batch_size
        self.n_input = num_input
        self.n_hidden = num_hidden
        self.n_classes = num_classes
        self.timestep = timestep
        self.name = 'CNN+RNN'
        self.x_train = tf.placeholder("float", [None, self.timestep, self.n_input]) # (?, timestep, n_input) 
        self.y_train = tf.placeholder("float", [None, self.timestep, self.n_classes]) # (?, timestep, n_class)
        self.seq_len = tf.placeholder(tf.int32, [None]) 
        self.n_hidden = 256

    def build(self):

        self.logits = self.CNNRNN(self.x_train) #(batch_size * timestep, n_classes)
        self.so_max = tf.nn.softmax(self.logits)    
        self.pred = tf.argmax(self.logits, 1) # (batch * timestep, 1) 
        self.y = tf.reshape(self.y_train, [-1, self.n_classes])

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, 
            labels=self.y))

        self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr, momentum=0.9).minimize(self.loss_op) 

    def CNNRNN(self, x):
        # x input (batch_size, timestep, n_input)
        with slim.arg_scope([slim.conv2d], padding='SAME', 
                activation_fn=swish_sigmoid,
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                weights_regularizer=slim.l2_regularizer(0.0005)):

            net = slim.conv2d(x, 128, [3], scope='conv1') 
            net = slim.conv2d(net, 256, [3], scope='conv2')

        net = slim.flatten(net, scope='flatten1')
        net = tf.reshape(net, [-1, self.timestep, self.n_hidden])

        lstm_fw_cell1 = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell1 = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)

        lstm_fw_cell2 = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell2 = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)

        dropout_lstm_fw1 = tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True), output_keep_prob=0.5)
        dropout_lstm_bw1 = tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True), output_keep_prob=0.5)

        dropout_lstm_fw2 = tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True), output_keep_prob=0.5)
        dropout_lstm_bw2 = tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True), output_keep_prob=0.5)


        stack_forward_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell1, dropout_lstm_fw1, lstm_fw_cell2, dropout_lstm_fw2], state_is_tuple=True)
        stack_backward_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell1, dropout_lstm_bw1, lstm_bw_cell2, dropout_lstm_fw2], state_is_tuple=True)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(stack_forward_cell, stack_backward_cell, net, 
            sequence_length=self.seq_len, dtype=tf.float32) 
        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, self.n_hidden*2])

        
        return slim.fully_connected(outputs, self.n_classes, activation_fn=None) # x output: (batch*timestep, n_hidden)

