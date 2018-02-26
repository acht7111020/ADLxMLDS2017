from agent_dir.agent import Agent
import numpy as np
import random
import tensorflow as tf
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from keras import backend as K 

EXPLORATION_STEPS = 1e6  # Number of exploration steps
INITIAL_EPSILON = 1.0  # Initial value of epsilon
FINAL_EPSILON = 0.05  # Final value of epsilon
INITIAL_REPLAY_SIZE = 1000
NUM_REPLAY_MEMORY = 10000  # Size of memory during training
NO_OP_STEPS = 30
TRAIN_INTERVAL = 4
TARGET_UPDATE_INTERVAL = 500

# Sample code from : https://github.com/hzm2016/DQN
class DQN(object):
    def __init__(self, state_size,
                       action_size,
                       savepath,
                       learning_rate = 0.0001,
                       minibatch_size = 32,
                       discount_factor = 0.99,
                       DoubleDQN = True,
                       DuelDQN = True,
                       train=True,
                       comment=''):

        if train:
            self.savepath = savepath + 'model.ckpt'
            path_args = savepath + 'output.log'
            self.writer = open(path_args, 'w')
            self.writer.write(comment)
            self.writer.write('\nstart training\n')

        self.num_actions = action_size
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / (EXPLORATION_STEPS)
        self.lr = learning_rate
        self.batch_size = minibatch_size
        self.gamma = discount_factor
        self.DoubleDQN = DoubleDQN
        self.DuelDQN = DuelDQN

        # Parameters used for summary
        self.total_reward, self.total_q_max, self.total_loss, self.duration, self.episode = 0, 0, 0, 0, 0 
        self.t = 0
        self.avg_score = []

        self.replay_memory = deque()

        self.q_network = self.build_network('q_net')
        q_network_weights = self.q_network.trainable_weights
        self.target_network = self.build_network('tar_net')
        target_network_weights = self.target_network.trainable_weights

        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        self.build_training_op(q_network_weights)

        # self.sess = tf.InteractiveSession()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver(q_network_weights, max_to_keep=20) 

        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.update_target_network)

    def build_network(self, name):
        with tf.name_scope(name):
            input_x = Input(shape=(84, 84, 4))

            net = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_x)
            net = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(net)
            net = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(net)
            net = Flatten()(net)
            net = Dense(512, activation='relu')(net)

            if self.DuelDQN:
                y = Dense(self.num_actions + 1, activation='linear')(net)
                Q = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(self.num_actions,))(y)
            else:
                Q = Dense(self.num_actions)(net)

            model = Model(inputs=input_x, outputs=Q)

        return model

    def build_training_op(self, q_network_weights): 
        self.state = tf.placeholder(tf.float32, (None, 84, 84, 4), name="state")
        self.q_values = tf.identity(self.q_network(self.state), name="q_values")
        self.predicted_actions = tf.argmax(self.q_values, axis=1, name="predicted_actions")

        self.next_state = tf.placeholder(tf.float32, (None, 84, 84, 4), name="next_state")
        self.next_state_mask = tf.placeholder(tf.float32, (None, ), name="next_state_mask") # 0 for terminal states
        self.rewards = tf.placeholder(tf.float32, (None, ), name="rewards")

        self.next_q_values_targetqnet = tf.stop_gradient(self.target_network(self.next_state), name="next_q_values_targetqnet")

        if self.DoubleDQN:
            print ("Double DQN")
            self.next_q_values_qnet = tf.stop_gradient(self.q_network(self.next_state), name="next_q_values_qnet")
            self.next_selected_actions = tf.argmax(self.next_q_values_qnet, axis=1)
            self.next_selected_actions_onehot = tf.one_hot(indices=self.next_selected_actions, depth=self.num_actions)
            self.next_max_q_values = tf.stop_gradient( tf.reduce_sum( 
                tf.multiply( self.next_q_values_targetqnet, self.next_selected_actions_onehot ), reduction_indices=[1,]) * self.next_state_mask )

        else:
            print ("Normal DQN")
            self.next_max_q_values = tf.reduce_max(self.next_q_values_targetqnet, reduction_indices=[1,]) * self.next_state_mask

        self.target_q_values = self.rewards + self.gamma * self.next_max_q_values

        with tf.name_scope("optimization_step"):
            self.action_mask = tf.placeholder(tf.float32, (None, self.num_actions) , name="action_mask") #action that was selected
            self.y = tf.reduce_sum( self.q_values * self.action_mask , reduction_indices=[1,])

            self.error = tf.abs(self.y - self.target_q_values)
            quadratic_part = tf.clip_by_value(self.error, 0.0, 1.0)
            linear_part = self.error - quadratic_part
            # self.loss = tf.reduce_mean( 0.5*tf.square(quadratic_part) + linear_part )
            self.loss = tf.reduce_mean( tf.sqrt(1+tf.square(quadratic_part)) - 1 + linear_part , axis=-1)

            self.train_op = tf.train.RMSPropOptimizer(self.lr, momentum=0.99, epsilon=0.01).minimize(self.loss, var_list=q_network_weights)

    def get_action(self, state):
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
            mode = 0
        else: 
            action = self.sess.run(self.predicted_actions, {self.state:[state] } )[0]
            mode = 1

        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action, mode

    def run(self, state, action, reward, terminal, next_state, action_mode):
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                q_values = self.train_network()
                
            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

        self.total_reward += reward

        state = np.expand_dims(state, axis=0)
        self.total_q_max += np.max(self.sess.run(self.q_values, feed_dict={self.state: state})) 

        self.duration += 1

        if terminal: 
            if len(self.avg_score) < 30:
                self.avg_score.append(self.total_reward)
            else:
                index = self.episode % 30 
                self.avg_score[index] = self.total_reward

            # Save network
            if self.episode > 10000 and self.episode % 500 == 0 : 
                save_path = self.saver.save(self.sess, self.savepath, global_step=self.episode)
                print('Successfully saved: ' + save_path)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'rand'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('Episode: {0:6d}, T: {1:8d}, Step: {2:5d}, Epsilon: {3:.5f}, Score: {4:3.0f}, Avg_score: {5:3.3f}, Avg_maxQ: {6:2.4f}, mode: {7}{8:d}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                np.mean(self.avg_score),
                mode, action_mode))
            self.writer.write('Episode: {0:6d}, T: {1:8d}, Step: {2:5d}, Epsilon: {3:.5f}, Score: {4:3.0f}, Avg_score: {5:3.3f}, Avg_maxQ: {6:2.4f}, mode: {7}{8:d}\n'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                np.mean(self.avg_score),
                mode, action_mode))

            self.total_reward, self.total_q_max, self.total_loss, self.duration = 0, 0, 0, 0 
            self.episode += 1

        self.t += 1

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = [] 

        # Sample random minibatch from replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            if data[4]:
                terminal_batch.append(0)
            else:
                terminal_batch.append(1)

        batch_actions = np.zeros( (self.batch_size, self.num_actions) )
        for i in range(self.batch_size):
            batch_actions[i, action_batch[i]] = 1
 
        q_values, _, loss = self.sess.run([self.q_values, self.train_op, self.loss], feed_dict={
            self.state: np.asarray(state_batch),
            self.next_state: np.asarray(next_state_batch),
            self.next_state_mask: np.asarray(terminal_batch),
            self.rewards: np.asarray(reward_batch),
            self.action_mask: batch_actions
            })

        self.total_loss += loss 
        return q_values

    def load_checkpoint(self, path):
        print("Loading checkpoint...")
        latest_ckpt = tf.train.latest_checkpoint(path)
        self.saver.restore(self.sess, latest_ckpt)

    def get_action_at_test(self, state): 
        state = np.expand_dims(state, axis=0)
        action = np.argmax(self.sess.run(self.q_values, feed_dict={self.state: state}))
        return action

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)
        tf.reset_default_graph()
        self.env = env
        self.savepath = args.savepath + ''
        self.state_size = (84, 84, 4)
        self.action_size = self.env.action_space.n

        self.model = DQN(
                state_size=self.state_size,
                action_size=self.action_size,
                savepath=self.savepath,
                comment=args.comment,
                train=args.train_dqn
            );

        if args.test_dqn:
            self.model.load_checkpoint(args.loadpath)
            print('loading trained model') 

    def init_game_setting(self):
        self.t = 0

    def train(self):
        train_t = 0
        while train_t < 43000: # max to train
            state = self.env.reset()
            done = False
            for _ in range(random.randint(1, NO_OP_STEPS)):
                state, reward, done, _ = self.env.step(0)  # Do nothing

            while not done:
                action, action_mode = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                # env.render()
                self.model.run(state, action, reward, done, next_state, action_mode)
                state = next_state

            train_t += 1

    def make_action(self, observation, test=True):
        # prevent infinite loop
        if self.t > 2500:
            action = random.randrange(self.action_size)
        else:
            action = self.model.get_action_at_test(observation)
        self.t += 1
        return action

