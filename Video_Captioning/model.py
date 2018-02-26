import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer 

rnn = tf.contrib.rnn
slim = tf.contrib.slim

class Seq2seq():
    def __init__(self, batch_size, learning_rate, vocabsize, emb_dim, frame_size, feat_dim, pretrained_wemb, trainable=True, pred_batch_size=1):
        self.lr = learning_rate
        self.bs = batch_size
        self.fs = frame_size
        self.caption_len = 41 # max len of train/test caption
        self.feat_dim = feat_dim    # 4095
        self.emb_dim = emb_dim
        self.vocabsize = vocabsize
        self.name = 'Seq2Seq'
        self.trainable = trainable

        if self.trainable == True:
            self.pred_batch_size = self.bs
        else:
            self.pred_batch_size = pred_batch_size
 
        self.wembed = tf.get_variable("word_embeddings", initializer=pretrained_wemb) # , trainable=False

        # add placeholder
        self.x = tf.placeholder(tf.float32, [self.pred_batch_size, self.fs, self.feat_dim])
        self.x_mask = tf.placeholder(tf.float32, [self.bs, self.fs])
        self.caption = tf.placeholder(tf.int32, [self.bs, self.caption_len])
        self.caption_mask = tf.placeholder(tf.float32, [self.bs, self.caption_len])
        self.prob_sch = tf.placeholder(tf.float32, [])
        self.roll = tf.placeholder(tf.float32, [])

        self.hist = tf.placeholder(tf.float32, [self.pred_batch_size, self.vocabsize])
        self.indices = tf.placeholder(tf.float32, [self.pred_batch_size, 1])
        self.seq_len = tf.placeholder(tf.int32, [self.bs])

        # RNN parameters
        self.n_hidden = 512

    def build_model(self):
        lstm1 = rnn.GRUCell(self.n_hidden )
        lstm2 = rnn.GRUCell(self.n_hidden )

        self.weights = {
            'V': tf.get_variable(name="V", shape=[1, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer()),
            'W1': tf.get_variable("W1",
                shape=[self.n_hidden, self.n_hidden],
                initializer=tf.contrib.layers.xavier_initializer()),
            'W2': tf.get_variable("W2",
                shape=[self.n_hidden, self.n_hidden],
                initializer=tf.contrib.layers.xavier_initializer()),
            'wemb': tf.get_variable("W_logits",
                shape=[self.n_hidden, self.vocabsize],
                initializer=tf.contrib.layers.xavier_initializer())
        }
        self.biases = {
            'wemb': tf.get_variable("b_logits",
                shape=[self.vocabsize],
                initializer=tf.constant_initializer(0))
        }

        video_flat = tf.reshape(self.x, [-1, self.feat_dim])
        # image_emb = tf.nn.xw_plus_b( video_flat, self.weights['hidden'], self.biases['hidden'] ) # (batch_size*fs, dim_hidden)
        image_emb = tf.layers.dense(video_flat, self.n_hidden, kernel_initializer=tf.contrib.layers.xavier_initializer())
        image_emb = tf.reshape(image_emb, [-1, self.fs, self.n_hidden]) # bs, fs, n_hidden

        state1 = lstm1.zero_state(self.pred_batch_size, tf.float32) # (batch_size, n_hidden) 
        state2 = lstm2.zero_state(self.pred_batch_size, tf.float32) # (batch_size, n_hidden) 
        padding1 = tf.zeros([self.pred_batch_size, self.n_hidden])
        padding2 = tf.zeros([self.pred_batch_size, self.emb_dim+self.vocabsize+1])

        self.loss_op = 0.0
        if self.trainable == False: # Test!
            # self.embeds = []
            self.generated_words = []
            self.probs = []

        hist = self.hist
        indices = self.indices

        ##############################  Encoding Stage ##################################
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in range(0, self.fs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = lstm1(image_emb[:,i,:], state1) 

                if self.trainable == True:
                    if i == 0: 
                        hidden1 = tf.expand_dims(output1, axis=1) # batch_size, 1, n_hidden
                    else:
                        hidden1 = tf.concat([hidden1, tf.expand_dims(output1, axis=1)], axis=1) # batch_size, fs, n_hidden
                else:
                    if i == 0:
                        hidden1 = output1 # n_hidden
                    else:
                        hidden1 = tf.concat([hidden1,output1], axis=0) # fs, n_hidden


                with tf.variable_scope("LSTM2"):
                    output2, state2 = lstm2(tf.concat(axis=1, values=[padding2, output1]), state2)
                    # output2 (batch_size, n_hidden)
                    # state2 (batch_size, n_hidden) * 2


        ############################# Decoding Stage ######################################
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in range(0, self.caption_len): ## Phase 2 => only generate captions

                with tf.device("/cpu:0"):
                    if i == 0: # trainable = True or False
                        current_embed = tf.nn.embedding_lookup(self.wembed, tf.zeros([self.pred_batch_size], dtype=tf.int64)) 
                    elif self.trainable == True:
                        def f1(): return tf.nn.embedding_lookup(self.wembed, self.caption[:, i-1])  # true-on
                        def f2(): return tf.nn.embedding_lookup(self.wembed, max_prob_index)

                        def f3(): return tf.cast(self.caption[:, i-1] , tf.int32)      # true-on
                        def f4(): return tf.cast(max_prob_index , tf.int32)
                        # roll = self.ep*np.random.rand()
                        current_embed = tf.cond(tf.less(self.roll, self.prob_sch), true_fn=f1, false_fn=f2)
                        hist_prob_index = tf.cond(tf.less(self.roll, self.prob_sch), true_fn=f3, false_fn=f4)

                        logits_onehot = tf.one_hot(hist_prob_index, self.vocabsize)
                        hist = tf.add(hist, logits_onehot)
                        # print(current_embed)
                    # wembed: vocab_size, emb_dim
                    # caption: array of int
                    # current_embed: (batch_size, emb_dim)

                if i >= 0:
                    tf.get_variable_scope().reuse_variables()

                # Attention ver2
                hidden1 = slim.dropout(hidden1, 0.75) # (batch_size, fs, n_hidden)
                encoder_hid = tf.reshape(hidden1, [-1, self.n_hidden]) # (batch_size*fs, n_hidden)

                attention1 = tf.matmul(encoder_hid, self.weights['W1']) # (batch_size*fs, n_hidden)

                alpha = tf.tile(tf.expand_dims(state1, axis=1), [1, self.fs, 1]) # copy 80 times -> (batch_size, frame, n_hidden)
                alpha = tf.reshape(alpha, [-1, self.n_hidden]) # (batch*frame, n_hidden)
                attention2 = tf.matmul(alpha, self.weights['W2'])  # (batch*frame, n_hidden)
                
                atten = self.weights['V'] * tf.tanh(attention1 + attention2) 
                context = tf.reshape(atten, [-1, self.fs, self.n_hidden])
                context = tf.nn.softmax(context) * hidden1 # bs, fs, n_hidden
                context = tf.reduce_sum(context, axis=1) # bs, n_hidden
                

                with tf.variable_scope("LSTM1"):
                    output1, state1 = lstm1(context, state1)

                with tf.variable_scope("LSTM2"):
                    current_embed = tf.cast(current_embed, tf.float32)
                    indices *= i
                    input2 = tf.concat(axis=1, values=[output1, current_embed, hist, indices])  # add hist information and index informaton
                    output2, state2 = lstm2(input2, state2)
                    

                if self.trainable == True: 
                    output2 = slim.dropout(output2, 0.75)

                logit_words = tf.nn.xw_plus_b(output2, self.weights['wemb'], self.biases['wemb']) # (batch_size, vocab_size) 
                
                if self.trainable == True:
                    labels = tf.reshape(self.caption[:, i], [-1]) # (batch_size)
                    loss_per_word = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logit_words, labels=labels)
                    loss_per_word = loss_per_word * self.caption_mask[:,i]
                    current_loss = tf.reduce_mean(loss_per_word)/self.bs

                    self.loss_op += current_loss

                    # for scheduleing !!
                    max_prob_index = tf.argmax(logit_words, 1)
                    
                else:
                    prob = tf.nn.softmax(logit_words) # probablitity of word
                    max_prob_index = tf.argmax(logit_words, 1)[0]
                    
                    self.generated_words.append(max_prob_index)
                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.wembed, max_prob_index)
                        current_embed = tf.expand_dims(current_embed, 0)
                    self.probs.append(prob)
                    # self.embeds.append(current_embed)
                    logits_onehot = tf.one_hot(max_prob_index, self.vocabsize)
                    hist = tf.add(hist, logits_onehot) # append this action to history action

                    

        if self.trainable == True:
            self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr, epsilon=1e-08, decay=0.0).minimize(self.loss_op)
            # self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_op)
