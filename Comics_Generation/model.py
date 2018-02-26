import numpy as np
import tensorflow as tf 

slim = tf.contrib.slim
layers = tf.contrib.layers

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# Sample Code From:
# https://github.com/paarthneekhara/text-to-image
# https://github.com/carpedm20/DCGAN-tensorflow

class GAN():
    def __init__(self, batch_size=64, img_size=96, y_dim=23, noise_dim=100, learning_rate=0.0001, trainable=True):
        self.lr = learning_rate
        self.bs = batch_size
        self.img_size = img_size
        self.y_dim = y_dim
        self.noise_dim = noise_dim
        self.trainable = trainable
        self.name = 'GAN'
        self.imgs = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3]) # 64 64 64 3
        self.labels = tf.placeholder(tf.float32, [None, self.y_dim]) # 64 2400
        self.noises = tf.placeholder(tf.float32, [None, self.noise_dim]) # 64 100

        # for cond gan
        self.imgs_wrong = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3]) # 64 64 64 3
        self.labels_wrong = tf.placeholder(tf.float32, [None, self.y_dim]) # 64 2400

    def build(self):

        self.generated_op = self.cond_generator_fn(self.noises, self.labels, train=self.trainable) #x_
        self.d_fake_output_op = self.cond_discriminator_fn(self.generated_op, self.labels) #d_
        self.d_real_output_op = self.cond_discriminator_fn(self.imgs, self.labels, reuse=True) #d
        self.d_wrong_img = self.cond_discriminator_fn(self.imgs_wrong, self.labels, reuse=True) #d
        self.d_wrong_label = self.cond_discriminator_fn(self.imgs, self.labels_wrong, reuse=True) #d 

        self.sampler = tf.identity(self.cond_generator_fn(self.noises, self.labels, reuse=True, train=False), name='sampler')

        self.g_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_output_op, labels=tf.ones_like(self.d_fake_output_op)))

        self.d_loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_real_output_op, labels=tf.ones_like(self.d_real_output_op))) \
                    + (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_fake_output_op, labels=tf.zeros_like(self.d_fake_output_op))) + \
                       tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_wrong_img, labels=tf.zeros_like(self.d_wrong_img))) +\
                       tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_wrong_label, labels=tf.zeros_like(self.d_wrong_label))) ) / 3

        vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'dis')
        vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen')
        
        self.d_train_op = tf.train.AdamOptimizer(self.lr, 0.5, 0.9).minimize(loss=self.d_loss_op, var_list=vars_d) #, epsilon=1e-08, decay=0.0

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.g_train_op = tf.train.AdamOptimizer(self.lr, 0.5, 0.9).minimize(loss=self.g_loss_op, var_list=vars_g)

    def cond_generator_fn(self, noise, label, train, weight_decay=2.5e-5, reuse=False):
        with tf.variable_scope('gen', reuse=reuse):
            with slim.arg_scope(
                [layers.fully_connected, layers.conv2d_transpose],
                activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                ):
                x = tf.concat([noise, label], axis=-1)
                net = tf.layers.batch_normalization(layers.fully_connected(x, 6 * 6 * 512), \
                    training=train, momentum=0.9, epsilon=1e-5)
                net = tf.reshape(net, [-1, 6, 6, 512])
                net = tf.nn.relu(tf.layers.batch_normalization( \
                    layers.conv2d_transpose(net, 256, [5, 5], stride=2), training=train, momentum=0.9, epsilon=1e-5))
                net = tf.nn.relu(tf.layers.batch_normalization( \
                    layers.conv2d_transpose(net, 128, [5, 5], stride=2), training=train, momentum=0.9, epsilon=1e-5))
                net = tf.nn.relu(tf.layers.batch_normalization( \
                    layers.conv2d_transpose(net, 64, [5, 5], stride=2), training=train, momentum=0.9, epsilon=1e-5))
                net = layers.conv2d_transpose(net, 3, [5, 5], stride=2, normalizer_fn=None, activation_fn=None) 
                net = tf.nn.tanh(net)
        return net

    def cond_discriminator_fn(self, img, label, weight_decay=2.5e-5, reuse=False):
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()
        with tf.variable_scope('dis', reuse=reuse):
            with slim.arg_scope(
                [layers.conv2d, layers.fully_connected],
                activation_fn=None, normalizer_fn=None,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.02)
                ):

                net = lrelu(layers.conv2d(img, 64, [5, 5], stride=2))
                # net = lrelu(layers.conv2d(net, 64, [5, 5], stride=1))
                net = lrelu(layers.conv2d(net, 128, [5, 5], stride=2))
                # net = lrelu(layers.conv2d(net, 128, [5, 5], stride=1))
                net = lrelu(layers.conv2d(net, 256, [5, 5], stride=2))
                # net = lrelu(layers.conv2d(net, 256, [5, 5], stride=1))
                net = lrelu(layers.conv2d(net, 512, [5, 5], stride=2))

                embed_y = tf.expand_dims(label, 1)
                embed_y = tf.expand_dims(embed_y, 2)
                tiled_embeddings = tf.tile(embed_y, [1,6,6,1])

                h3_concat = tf.concat( [net, tiled_embeddings], axis=-1)

                net = lrelu(layers.conv2d(h3_concat, 512, [1, 1], stride=1, padding='valid'))
                net = layers.flatten(net)
                # net = layers.fully_connected(net, 4*4*512)
                net = layers.fully_connected(net, 1, normalizer_fn=None, activation_fn=None)
        return net

