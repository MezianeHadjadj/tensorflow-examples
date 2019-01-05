import tensorflow as tf
import gym
import random
import numpy as np
import math
import matplotlib.pyplot as plt


class TrainingModel:
    def __init__(self, num_states, num_actions, batch_size):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        # define the placeholders
        self.states = None
        self.actions = None
        # the output operations
        self.logits = None
        self.optimizer = None
        # now setup the algo structure
        self.training_structure()

    def training_structure(self):
        self.states = tf.placeholder(shape=[None, self.num_states], dtype=tf.float32)
        self.q_learning_s_a = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        layer_1 = tf.layers.dense(self.states, 50, activation=tf.nn.relu)
        layer_2 = tf.layers.dense(layer_1, 50, activation=tf.nn.relu)
        self.logits = tf.layers.dense(layer_2, self.num_actions)
        loss_function = tf.losses.mean_squared_error(self.q_learning_s_a, self.logits)
        self.optimizer = tf.train.AdamOptimizer().minimize(loss_function)

    def predict_one_action(self, state, s):
        return s.run(self.logits, feed_dict={self.states:
                                                     state.reshape(1, self.num_states)})

    def predict_batch_of_actions(self, states, s):
        return s.run(self.logits, feed_dict={self.states: states})

    def train_batch(self, s, x_batch, y_batch):
        s.run(self.optimizer, feed_dict={self.states: x_batch, self.q_learning_s_a: y_batch})


class StorageAdapter:
    def __init__(self):
        self.results = []

    def add_step_results(self, result):
        self.results.append(result)

    def fetch_results(self, nbr_results):
        if nbr_results > len(self.results):
            return random.sample(self.results, len(self.results))
        else:
            return random.sample(self.results, nbr_results)


class AgentTraining:
    def __init__(self, s, model, env, storageAdapter, decay):
        self.s = s
        self.env = env
        self.model = model
        self.storageAdapter = storageAdapter
        self.max_epsilon = 0.9
        self.min_epsilon = 0.1
        self.epsilon = 0.9
        self.decay = decay
        self.steps = 0
        self.reward_store = []
        self.max_x_store = []

    def run(self):
        state = self.env.reset()
        total_reward = 0
        max_x = -100
        while True:
            self.env.render()
            action = self.choose_action(state)
            next_state, reward, done, info = self.env.step(action)
            if next_state[0] >= 0.1:
                reward += 10
            elif next_state[0] >= 0.25:
                reward += 20
            elif next_state[0] >= 0.5:
                reward += 100

            if next_state[0] > max_x:
                max_x = next_state[0]
            # set next_state to None for storage sake if training is complete
            if done:
                next_state = None

            self.storageAdapter.add_step_results((state, action, reward, next_state))
            self.train()

            # exponentially decay the epsilon value
            self.steps += 1
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) \
                                      * math.exp(-_lambda * self.steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            total_reward += reward

            # if the training is done, exit the loop
            if done:
                self.reward_store.append(total_reward)
                self.max_x_store.append(max_x)
                break

        print("Step {}, Total reward: {}, Epsilon: {}".format(self.steps, total_reward, self.epsilon))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.model.num_actions - 1)
        else:
            return np.argmax(self.model.predict_one_action(state, self.s))


    def train(self):
        batch = self.storageAdapter.fetch_results(self.model.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self.model.num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_learning_s_a = self.model.predict_batch_of_actions(states, self.s)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        next_q_learning_s_a = self.model.predict_batch_of_actions(next_states, self.s)
        # setup training arrays
        x = np.zeros((len(batch), self.model.num_states))
        y = np.zeros((len(batch), self.model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_learning_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the training completed after action, so there is no max Q(s',a') prediction possible
                current_q[action] = reward
            else:
                GAMMA = 0.9
                current_q[action] = reward + GAMMA * np.amax(next_q_learning_s_a[i])
            x[i] = state
            y[i] = current_q
        self.model.train_batch(self.s, x, y)



if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    batch_size = 50
    _lambda= 1/8000
    num_states = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.n

    model = TrainingModel(num_states, num_actions, batch_size)
    storage = StorageAdapter()

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())
        gr = AgentTraining(s, model, env, storage,
                        _lambda)
        num_episodes = 300
        cnt = 0
        while cnt < num_episodes:
            if cnt % 10 == 0:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
            gr.run()
            cnt += 1
        plt.plot(gr.reward_store)
        plt.show()
        plt.close("all")
        plt.plot(gr.max_x_store)
        plt.show()
