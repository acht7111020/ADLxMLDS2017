from agent_dir.agent import Agent
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers import Conv2D
import tensorflow as tf
from keras import backend as K
import time
from scipy.misc import imresize

class Agent_PG(Agent):
    def __init__(self, env, args): 
        super(Agent_PG,self).__init__(env)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.Session(config=config)

        self.env = env
        # self.state_size = 80 * 80
        self.state_size = (80, 80, 1)
        self.action_size = 3
        print(self.action_size)
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()

        if args.retrain:
            self.load(args.loadpath)
            print('loading trained model')

        if args.test_pg:
            #you can load your model here
            self.load(args.loadpath)
            print('loading trained model')
        else:
            path_args = args.savepath + 'output.log'
            self.writer = open(path_args, 'w')

            self.writer.write(args.comment)
            self.writer.write('\nstart training\n')

            self.savepath = args.savepath
        
    def _build_model(self):
        model = Sequential()
        # model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        model.add(Conv2D(32, (8, 8), input_shape=(80, 80, 1), strides=(4, 4), padding='same', # data_format='channels_first',
                                activation='relu', kernel_initializer='truncated_normal'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same',# data_format='channels_first',
                                activation='relu', kernel_initializer='truncated_normal'))
        model.add(Flatten())
        # model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt) # 'categorical_crossentropy'
        return model

    def init_game_setting(self): 
        self.prev_x = None
        self.probs = []

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        # y = action
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self): 
        state = self.env.reset() # 210, 160, 3
        print(state.shape)
        prev_x = None
        score = 0
        avg_score = []
        episode = 0
        totalaction = 0
        while True:
            
            cur_x = my_preprocess(state)
            x = cur_x - prev_x if prev_x is not None else np.zeros(self.state_size)
            prev_x = cur_x

            gym_action, action, prob = self.act(x)
            # print(gym_action)
            state, reward, done, info = self.env.step(gym_action)
            score += reward
            totalaction += 1
            self.remember(x, action, prob, reward)

            if done:
                
                # calculate moving average
                if len(avg_score) < 30:
                    avg_score.append(score)
                else:
                    index = episode % 30 
                    avg_score[index] = score

                episode += 1
                # if episode % 10 == 0:
                loss = self.train_batch()
                # loss = np.mean(hist.history['loss'])
                this_score = np.mean(avg_score)

                print('Episode: %5d, Score: %2.0f, Average Score: %2.3f, Actions: %5d, Loss: %.4g ' % (episode, score, this_score, totalaction, loss))
                # print info
                self.writer.write('Episode: %5d, Score: %2.0f, Average Score: %2.3f, Loss: %.4g \n' % (episode, score, this_score, loss))

                # reset game setting
                score = 0
                totalaction = 0
                state = self.env.reset()
                prev_x = None
 

                # save model
                if episode > 0 and episode % 200 == 0 or episode == 1:
                    path = self.savepath + 'pong_' + str(episode) + '.h5'
                    self.save(path)
        

    def train_batch(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        gradients *= rewards

        X = (np.vstack([self.states]))
        Y = self.probs + self.learning_rate * (np.vstack([gradients]))

        scores = self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []
        return scores

    def act(self, state):
        # state = state.reshape([1, state.shape[0]])
        state = np.expand_dims(state, axis=0)
        # print(state.shape)
        aprob = self.model.predict(state, batch_size=1)[0]
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        # print(self.model.predict(state, batch_size=1))
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        # gym_action = action

        return action+1, action, prob

    def make_action(self, observation, test=True): 

        cur_x = my_preprocess(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.state_size)
        self.prev_x = cur_x

        action = self.act_test(x) 
        return action 

    def act_test(self, state): 
        state = np.expand_dims(state, axis=0) 
        aprob = self.model.predict(state, batch_size=1)[0]
        action = np.argmax(aprob)
        return action+1

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def my_preprocess(o):  # 80*80  
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = imresize(y, (80, 80)) 
    return np.expand_dims(resized.astype(np.float32), axis=2)
