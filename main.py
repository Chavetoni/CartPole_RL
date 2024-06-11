import os
import random
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop

# Ensure TensorFlow uses GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("Using GPU")
else:
    print("Using CPU")

def OurModel(input_shape, action_space):
    X_input = Input(input_shape)

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class DQNAgent:
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode = 'human')
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODES = 90
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.train_start = 90

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)

        self.scores = []
        self.epsilon_values = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = np.array(minibatch[i][0])
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = np.array(minibatch[i][3])
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        weights_file = f"{name}.h5"
        if os.path.exists(weights_file):
            self.model.load_weights(weights_file)
        else:
            print(f"File not found: {weights_file}")

    def save(self, name):
        self.model.save(f"{name}.h5")
            
    
    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = state[0] if isinstance(state, tuple) else state  # Extract the actual state array from the tuple if necessary
            state = np.array(state)
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                if e % 10 == 0:  # Render every 10 episodes to reduce rendering overhead
                    self.env.render()
                action = self.act(state)
                step_result = self.env.step(action)
                next_state = step_result[0]  # Extract next_state from the step result
                reward = step_result[1]
                done = step_result[2]
                next_state = np.array(next_state)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done: 
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.EPISODES, i, self.epsilon))
                    self.scores.append(i)
                    self.epsilon_values.append(self.epsilon)                  
                    if i == 200:
                        print("Saving trained model as cartpole-dqn.h5")
                        self.save("cartpole-dqn")
                        return
                self.replay()

    def test(self):
        self.load("cartpole-dqn")
        self.test_scores = []
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = state[0] if isinstance(state, tuple) else state  # Extract the actual state array from the tuple if necessary
            state = np.array(state)
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                if e % 10 == 0:  # Render every 10 episodes to reduce rendering overhead
                    self.env.render()
                action = np.argmax(self.model.predict(state))
                step_result = self.env.step(action)
                next_state = step_result[0]
                reward = step_result[1]
                done = step_result[2]
                next_state = np.array(next_state)
                next_state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    self.test_scores.append(i)
                    break
                state = next_state

    def plot_training(self):
        plt.figure(figsize=(12,5))
        plt.subplot(121)
        plt.plot(self.scores)
        plt.xlabel('Episodes')
        plt.ylabel('Scores')
        plt.title('Scores per Episode')

        plt.subplot(122)
        plt.plot(self.epsilon_values)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay')

        plt.show()

    def plot_moving_average(self, window = 10):
        scores_series = pd.Series(self.scores)
        moving_avg = scores_series.rolling(window=window).mean()

        plt.figure(figsize=(12.5))
        sns.lineplot(data=scores_series, labe = 'Scores')
        sns.lineplot(data=moving_avg, label="Moving Average")
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Scores and Moving Average')
        plt.show()


if __name__ == "__main__":
    agent = DQNAgent()
    # agent.run()
    agent.test()
    agent.plot_training()
    agent.plot_moving_average()
