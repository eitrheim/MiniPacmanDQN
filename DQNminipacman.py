# Using DQN strategy that takes an observation of PacmanGame and returns an action (integer from 1 to 9).
# Specific game parameters (such as number of monsters, diamonds, etc) stored in test_params.json.


import random
import numpy as np
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
import json

# load game and its parameters
from mini_pacman import PacmanGame
with open('test_params.json', 'r') as file:
    read_params = json.load(file)
game_params = read_params['params']
env = PacmanGame(**game_params)


# get current state as a vector of features
def get_state(obs):
    v = []
    x,y = obs['player']
    v.append(x)
    v.append(y)
    for x, y in obs['monsters']:
        v.append(x)
        v.append(y)
    for x, y in obs['diamonds']:
        v.append(x)
        v.append(y)
    for x, y in obs['walls']:
        v.append(x)
        v.append(y)
    return v


# constructing DQN
def create_dqn_model(input_shape, nb_actions):
    model = Sequential()
    model.add(Dense(units=64, input_shape=input_shape, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


obs=env.reset()
input_shape = (len(get_state(obs)),)
moves = [1,2,3,4,5,6,7,8,9]
nb_actions = len(moves)


from collections import deque
replay_memory_maxlen = 1_000_000 #storage of experienced transitions
replay_memory = deque([], maxlen=replay_memory_maxlen)


def epsilon_greedy(q_values, epsilon, n_outputs):
    if random.random() < epsilon:
        return random.randrange(n_outputs)  # random action
    else:
        return np.argmax(q_values)  # q-optimal action
    
    
n_steps = 500_000 # number of times we adjust weights
warmup = 10_000 # first iterations after random initiation before training starts
training_interval = 5 # number of steps after which (we adjust weights) dqn is retrained
copy_steps = 10_000 # number of steps after which weights of online network copied into target network
gamma = 0.99 # discount rate
batch_size = 64 # size of batch from replay memory 
eps_max = 1.0 # parameters of decaying sequence of eps
eps_min = 0.05 # run it down to 5%
eps_decay_steps = int(n_steps/2)
learning_rate = 0.001


online_network = create_dqn_model(input_shape, nb_actions)
online_network.compile(optimizer=Adam(learning_rate), loss='mse')
target_network = clone_model(online_network)
target_network.set_weights(online_network.get_weights())


# to visualize the online network 
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# print(online_network.summary())
# SVG(model_to_dot(online_network).create(prog='dot', format='svg'))
# from keras.utils import plot_model
# plot_model(online_network, to_file='online_network.png',show_shapes=True,show_layer_names=True)


# training the model
step = 0 # our start
iteration = 0
done = True # we are still alive
while step < n_steps:
    if done:
        obs = env.reset()
    iteration += 1
    q_values = online_network.predict(np.array([get_state(obs)]))[0]  
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    action = epsilon_greedy(q_values, epsilon, nb_actions)
    next_obs = env.make_action(action+1)
    reward = next_obs["reward"]
    done = next_obs["end_game"]
    replay_memory.append((obs, action, reward, next_obs, done))
    obs = next_obs

    if iteration >= warmup and iteration % training_interval == 0:
        step += 1
        minibatch = random.sample(replay_memory, batch_size)
        replay_state = np.array([get_state(x[0]) for x in minibatch])
        replay_action = np.array([x[1] for x in minibatch])
        replay_rewards = np.array([x[2] for x in minibatch])
        replay_next_state = np.array([get_state(x[3]) for x in minibatch])
        replay_done = np.array([x[4] for x in minibatch], dtype=int)
        target_for_action = replay_rewards + (1-replay_done) * gamma * \
                                    np.amax(target_network.predict(replay_next_state), axis=1)
        target = online_network.predict(replay_state)  # targets coincide with predictions ...
        target[np.arange(batch_size), replay_action] = target_for_action  #...except for targets with actions from replay
        online_network.fit(replay_state, target, epochs=step, verbose=1, initial_epoch=step-1)
        if step % copy_steps == 0:
            target_network.set_weights(online_network.get_weights())

            
# save the deep neural network that estimates the Q-values
online_network.save('saved_dqn_mini_pacman_model.h5')


from keras.models import load_model
dqn_model = load_model('saved_dqn_mini_pacman_model.h5')


def dqn_strategy(obs):
    q_values = dqn_model.predict(np.array([get_state(obs)]))[0]
    action = epsilon_greedy(q_values, eps_min, nb_actions)
    return action + 1


# Some sub-optimal strategies are availiable for comparison: random_strategy moves the agent by random selection of actions, naive_strategy uses some basic heuristics.
from mini_pacman import test, random_strategy, naive_strategy
test(strategy=random_strategy, log_file='test_pacman_log.json')
test(strategy=naive_strategy, log_file='test_pacman_log.json')
test(strategy=dqn_strategy, log_file='test_pacman_log.json')


# to see the game get played with the DQN strategy
import time
obs = env.reset()
env.render()
state = get_state(obs)
while not obs['end_game']:
    time.sleep(0.1)
    # select best next action using Q-Learning (no random component here, eps=0)
    action = dqn_strategy(obs)
    obs = env.make_action(action)
    state = get_state(obs)
    env.render()
print('Total score = {}'.format(obs['total_score']))
