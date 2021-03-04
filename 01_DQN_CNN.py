"""
https://towardsdatascience.com/playing-connect-4-with-deep-q-learning-76271ed663ca

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
import random

from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)

env.run(['random', 'random'])
env.render(mode="ipython", width=600, height=500, header=False)


def check_if_done(observation):
    done = [False, 'No Winner Yet']
    # horizontal check
    for i in range(6):
        for j in range(4):
            if observation[i][j] == observation[i][j + 1] == observation[i][j + 2] == observation[i][j + 3] == 1:
                done = [True, 'Player 1 Wins Horizontal']
            if observation[i][j] == observation[i][j + 1] == observation[i][j + 2] == observation[i][j + 3] == 2:
                done = [True, 'Player 2 Wins Horizontal']
    # vertical check
    for j in range(7):
        for i in range(3):
            if observation[i][j] == observation[i + 1][j] == observation[i + 2][j] == observation[i + 3][j] == 1:
                done = [True, 'Player 1 Wins Vertical']
            if observation[i][j] == observation[i + 1][j] == observation[i + 2][j] == observation[i + 3][j] == 2:
                done = [True, 'Player 2 Wins Vertical']
    # diagonal check top left to bottom right
    for row in range(3):
        for col in range(4):
            if observation[row][col] == observation[row + 1][col + 1] == observation[row + 2][col + 2] == observation[row + 3][col + 3] == 1:
                done = [True, 'Player 1 Wins Diagonal']
            if observation[row][col] == observation[row + 1][col + 1] == observation[row + 2][col + 2] == observation[row + 3][col + 3] == 2:
                done = [True, 'Player 2 Wins Diagonal']

    # diagonal check bottom left to top right
    for row in range(5, 2, -1):
        for col in range(3):
            if observation[row][col] == observation[row - 1][col + 1] == observation[row - 2][col + 2] == observation[row - 3][col + 3] == 1:
                done = [True, 'Player 1 Wins Diagonal']
            if observation[row][col] == observation[row - 1][col + 1] == observation[row - 2][col + 2] == observation[row - 3][col + 3] == 2:
                done = [True, 'Player 2 Wins Diagonal']
    return done


def create_model(type='dense'):
    if type == 'dense__':
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(7))

    elif type == 'cnn':
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(7, 7, 1)))
        # model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(8, (5, 5), padding='same', activation='relu'))
        # model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(4, (7, 7), padding='same', activation='relu'))
        # model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        # model.add(Dense(32, activation='relu'))
        model.add(Dense(7))

    return model


def compute_loss(logits, actions, rewards):
    neg_logprob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_logprob * rewards)
    return loss


def train_step(model, optimizer, observations, actions, rewards):
    with tf.GradientTape() as tape:
        # Forward propagate through the agent network

        logits = model(observations)
        loss = compute_loss(logits, actions, rewards)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


def get_action(model, observation, epsilon):
    # determine whether model action or random action based on epsilon
    act = np.random.choice(['model', 'random'], 1, p=[1 - epsilon, epsilon])[0]
    observation = np.array(observation).reshape(1, 6, 7, 1)
    observation = np.pad(observation, ((0, 0), (1, 0), (0, 0), (0, 0)), 'constant', constant_values=0)  # 1x7x7x1 symmetric for CNN
    # print('observations:', observation, observation.shape)
    logits = model.predict(observation)
    prob_weights = tf.nn.softmax(logits).numpy()

    if act == 'model':
        action = list(prob_weights[0]).index(max(prob_weights[0]))
    if act == 'random':
        action = np.random.choice(7)

    return action, prob_weights[0]


def check_if_action_valid(obs, action):
    if obs[action] == 0:

        valid = True
    else:
        valid = False
    return valid


def player_1_agent(observation, configuration):
    action, prob_weights = get_action(player_1_model, observation['board'], 0)
    if check_if_action_valid(observation['board'], action):
        return action
    else:
        valid_moves = [col for col in range(configuration.columns) if observation.board[col] == 0]
        action = random.choice(valid_moves)

    return action


class Memory:
    def __init__(self):
        self.clear()

    # Resets/restarts the memory buffer
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.info = []

    def add_to_memory(self, new_observation, new_action, new_reward):
        self.observations.append(new_observation)
        self.actions.append(new_action)
        self.rewards.append(float(new_reward))


def random_agent(observation, config):
    valid_moves = [col for col in range(config.columns) if observation.board[col] == 0]
    move = random.choice(valid_moves)
    return move

LEARNING_RATE = 0.01
EPISODES = 1
switch_to_negamax = 300
player_1_model = create_model(type='cnn')
player_1_model.summary()
# player_2_model = create_model()
win_count = 0

# train player 1 against random agent
# tf.keras.backend.set_floatx('float64')
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

env = make("connectx", debug=True)
memory = Memory()
epsilon = 1

for i_episode in range(EPISODES):
    if i_episode % 10 == 0:
        print('EPisode:', i_episode, 'win count:', win_count)
    if i_episode >= switch_to_negamax:
        trainer = env.train([None, 'negamax'])
    else:
        trainer = env.train([None, 'random'])

    observation = trainer.reset()['board']
    memory.clear()
    epsilon = epsilon * .99985
    overflow = False
    while True:
        action, _ = get_action(player_1_model, observation, epsilon)
        next_observation, dummy, overflow, info = trainer.step(action)
        observation = next_observation['board']
        observation = [float(i) for i in observation]
        done = check_if_done(np.array(observation).reshape(6, 7))

        # -----Customize Rewards Here------
        if done[0] == False:
            reward = [0.2, 0.4, 0.6, 1, 0.6, 0.4, 0.2][action]
        if 'Player 2' in done[1]:
            reward = -100
            print('lose')
        if 'Player 1' in done[1]:
            win_count += 1
            reward = 100
            print('win')
        if overflow == True and done[0] == False:
            reward = -1000
            done[0] = True
        # -----Customize Rewards Here------

        observation_tmp = np.array(observation).reshape(6, 7, 1)
        observation_tmp = np.pad(observation_tmp, ((1, 0), (0, 0), (0, 0)), 'constant', constant_values=0)  # 1x7x7x1 symmetric for CNN
        memory.add_to_memory(observation_tmp, action, reward)
        if done[0]:
            # train after each game

            train_step(player_1_model, optimizer,
                       observations=np.array(memory.observations),
                       actions=np.array(memory.actions),
                       rewards=memory.rewards)

            break

checkpoint_path = "models/DQN/dqn_CNN_2.ckpt"
player_1_model.save_weights(checkpoint_path)
# player_1_model.load_weights(checkpoint_path)

def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}

    # Agent 1 goes first (roughly) half the time
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds // 2)

    # Agent 2 goes first (roughly) half the time
    outcomes += [[b, a] for [a, b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds - n_rounds // 2)]

    print("Agent 1 Win Percentage:", np.round(outcomes.count([1, -1]) / len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1, 1]) / len(outcomes), 2))
    print("Draw Percentage:", np.round(outcomes.count([0, 0]) / len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))


print('Eval against random')
get_win_percentages(player_1_agent, 'random', n_rounds=50)
print('Eval against negamax')
get_win_percentages(player_1_agent, 'negamax', n_rounds=10)
