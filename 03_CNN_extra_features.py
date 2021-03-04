"""
https://towardsdatascience.com/playing-connect-4-with-deep-q-learning-76271ed663ca

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
import random
import collections

from kaggle_environments import evaluate, make, utils
from evaluation import get_win_percentages

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


def check_window(window, num_discs, piece, config):
    return (window.count(piece) == num_discs and window.count(0) == config.inarow-num_discs)


def count_windows(grid, num_discs, piece, config):
    num_windows = 0
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            if check_window(window, num_discs, piece, config):
                num_windows += 1
    return num_windows



def create_model(type='dense'):
    if type == 'dense__':
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        # model.add(Dense(50, activation='relu'))
        # model.add(Dense(50, activation='relu'))
        # model.add(Dense(50, activation='relu'))
        # model.add(Dense(50, activation='relu'))
        model.add(Dense(7))

    elif type == 'cnn':
        model = Sequential()
        model.add(Conv3D(16, (3, 4, 4), padding='same', activation='relu', input_shape=(7, 4, 4, 1)))
        # model.add(MaxPooling2D((2, 2)))
        model.add(Conv3D(8, (5, 4, 4), padding='same', activation='relu'))
        # model.add(MaxPooling2D((2, 2)))
        model.add(Conv3D(4, (7, 4, 4), padding='same', activation='relu'))
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


def get_action(model, observation, epsilon, config):
    grid = np.asarray(observation.board).reshape(config.rows, config.columns)
    extra_features = featureSpace(grid, config, observation.mark)
    extra_features = np.array([col['matrix'] for col in extra_features])

    # determine whether model action or random action based on epsilon
    act = np.random.choice(['model', 'random'], 1, p=[1 - epsilon, epsilon])[0]
    observation = np.array(extra_features).reshape(1, 7, 4, 4, 1)
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
    action, prob_weights = get_action(player_1_model, observation, 0, configuration)
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


def coins_in_a_row(window, mark):
    # l = np.where(np.array(window) == (obs.mark % 2) + 1)[0]
    n = 0
    for value in window:
        if value == mark:
            n += 1
        else:
            return n
    return n


def featureSpace(grid, config, mark):
    feature = []
    for col in range(config.columns):
        feature.append({})
        feature_per_pos = np.zeros([4, 4])  # (left/up, right/down, overall, win) x (horizontal, diagonal_down, vertical, diagonal_up)

        n_coins_in_col = ((grid[:, 6] == 1) | (grid[:, 6] == 2)).sum()
        if n_coins_in_col == config.rows:  # row is already full
            feature[col]['full'] = 1
            feature[col]['matrix'] = np.zeros([4, 4]) - 1
            feature[col]['maxRow'] = -1
            feature[col]['win'] = -1
        else:
            feature[col]['full'] = 0
            row = n_coins_in_col  # new empty row in this column

            ### horizontal
            # left
            l2 = list(range(max(min(col - 1, config.columns), -1), max(min(col - 1 - config.inarow, config.columns), -1), -1))
            l1 = [row] * len(l2)
            n = min(len(l1), len(l2))
            feature_per_pos[0, 0] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], mark)

            # right
            l2 = list(range(max(min(col + 1, config.columns), -1), max(min(col + 1 + config.inarow, config.columns), -1)))
            l1 = [row] * len(l2)
            n = min(len(l1), len(l2))
            feature_per_pos[1, 0] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], mark)

            # overall
            feature_per_pos[2, 0] = feature_per_pos[0, 0] + feature_per_pos[1, 0]
            # win move
            feature_per_pos[3, 0] = int(feature_per_pos[2, 0] + 1 >= config.inarow)

            ### vertical
            # up (darüber)
            feature_per_pos[0, 2] = 0

            # down (darunter)
            l1 = list(range(max(min(row - 1, config.rows), -1), max(min(row - 1 - config.inarow, config.rows), -1), -1))
            l2 = [col] * len(l1)
            n = min(len(l1), len(l2))
            feature_per_pos[1, 2] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], mark)

            # overall
            feature_per_pos[2, 2] = feature_per_pos[0, 2] + feature_per_pos[1, 2]
            # win move
            feature_per_pos[3, 2] = int(feature_per_pos[2, 2] + 1 >= config.inarow)

            ### diagonal down
            # left (links oben)
            l1 = list(range(max(min(row + 1, config.rows), -1), max(min(row + 1 + config.inarow, config.rows), -1)))
            l2 = list(range(max(min(col - 1, config.columns), -1), max(min(col - 1 - config.inarow, config.columns), -1), -1))
            n = min(len(l1), len(l2))
            feature_per_pos[0, 1] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], mark)

            # right (rechts unten)
            l1 = list(range(max(min(row - 1, config.rows), -1), max(min(row - 1 - config.inarow, config.rows), -1), -1))
            l2 = list(range(max(min(col + 1, config.columns), -1), max(min(col + 1 + config.inarow, config.columns), -1)))
            n = min(len(l1), len(l2))
            feature_per_pos[1, 1] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], mark)

            # overall
            feature_per_pos[2, 1] = feature_per_pos[0, 1] + feature_per_pos[1, 1]
            # win move
            feature_per_pos[3, 1] = int(feature_per_pos[2, 1] + 1 >= config.inarow)

            ### diagonal up
            # left (links unten)
            l1 = list(range(max(min(row - 1, config.rows), -1), max(min(row - 1 - config.inarow, config.rows), -1), -1))
            l2 = list(range(max(min(col - 1, config.columns), -1), max(min(col - 1 - config.inarow, config.columns), -1), -1))
            n = min(len(l1), len(l2))
            feature_per_pos[0, 3] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], mark)

            # right (rechts oben)
            l1 = list(range(max(min(row + 1, config.rows), -1), max(min(row + 1 + config.inarow, config.rows), -1)))
            l2 = list(range(max(min(col + 1, config.columns), -1), max(min(col + 1 + config.inarow, config.columns), -1)))
            n = min(len(l1), len(l2))
            feature_per_pos[1, 3] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], mark)

            # overall
            feature_per_pos[2, 3] = feature_per_pos[0, 3] + feature_per_pos[1, 3]
            # win move
            feature_per_pos[3, 3] = int(feature_per_pos[2, 1] + 1 >= config.inarow)

            feature[col]['matrix'] = feature_per_pos
            feature[col]['maxRow'] = feature_per_pos[2, :].max()
            feature[col]['win'] = feature_per_pos[3, :].sum() > 0

    return feature


def get_reward(done, action, observation_old, observation_next, mark, silent=False):

    add_threes = count_windows(np.asarray(observation_next).reshape(6, 7), 3, mark, config) - count_windows(np.asarray(observation_old).reshape(6, 7), 3, mark, config)
    add_threes_opp = count_windows(np.asarray(observation_next).reshape(6, 7), 3, mark % 2 + 1, config) - count_windows(np.asarray(observation_old).reshape(6, 7), 3, mark % 2 + 1, config)
    win = 0

    if done[0] == False:

        if overflow == True:
            reward = -1000
            done[0] = True
        else:
            reward = [0.2, 0.4, 0.6, 1, 0.6, 0.4, 0.2][action]
            if add_threes:
                reward += 10
            if add_threes_opp:
                reward += -5

    elif 'Player 2' in done[1]:
        reward = -100
        print('lose') if not silent else None

    elif 'Player 1' in done[1]:
        win = 1
        reward = 100
        print('win') if not silent else None

    return reward, win


def drop_piece(grid, col, mark, config):
    next_grid = grid.copy()
    for row in range(config.rows - 1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid


def step_andCounterStep(action, grid, opp_agent, mark, config, observation_template):
    # agent's move
    child_grid_1 = drop_piece(grid, action, mark, config)
    # opponents move
    observation_template.board = list(child_grid_1.flatten())
    action_opp = opp_agent(observation_template, config)
    if child_grid_1.flatten()[action_opp] != 0:
        print('invalid move to %d' % action_opp)
    child_grid_2 = drop_piece(child_grid_1, action_opp, (mark % 2) + 1, config)
    return child_grid_2


LEARNING_RATE = 0.01
EPISODES = 2000
first_n_random = 1000
switch_to_negamax = 1000000
player_1_model = create_model(type='cnn')
player_1_model.build((None, 7, 4, 4, 1))
player_1_model.summary()
# player_2_model = create_model()
win_count = 0

# train player 1 against random agent
# tf.keras.backend.set_floatx('float64')
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

env = make("connectx", debug=True)
memory = Memory()
epsilon = 1
GameConfig = collections.namedtuple('GameConfig', ['rows', 'columns', 'inarow'])
config = GameConfig(rows=6, columns=7, inarow=4)

trainer = env.train([None, 'random'])
observation_full = trainer.reset()

for i_episode in range(EPISODES):
    if i_episode % 10 == 0:
        print('Episode:', i_episode, 'win count:', win_count)

    if i_episode >= switch_to_negamax:      # against good player
        opp_agent = env.agents['negamax']
    elif i_episode < first_n_random:        # against random player
        opp_agent = env.agents['random']
    else:                                   # against itself
        opp_agent = player_1_agent

    observation = observation_full['board']
    observation = [0] * config.rows * config.columns
    mark = observation_full.mark
    mark_opp = (observation_full.mark % 2) + 1
    memory.clear()
    epsilon = epsilon * .99985
    overflow = False
    step_counter = 0
    while True:
        board_old = observation
        valid_moves = [col for col in range(config.columns) if board_old[col] == 0]

        # collect and store all possible next moves
        for action in valid_moves:
            # step
            grid = np.array(board_old).reshape(config.rows, config.columns)
            grid_next = step_andCounterStep(action, grid, opp_agent, mark, config, observation_full)

            # rewards
            done = check_if_done(grid_next)
            board_next = list(grid_next.flatten())
            reward, win = get_reward(done, action, board_old, board_next, mark, silent=True)

            # add to memory
            extra_features = featureSpace(grid_next, config, mark)
            extra_features = np.array([c['matrix'] for c in extra_features])
            extra_features = np.array(extra_features).reshape(7, 4, 4, 1)
            memory.add_to_memory(extra_features, action, reward)

        # agent 1 move
        # action = random.choice(valid_moves)
        observation_full.board = board_old
        action, _ = get_action(player_1_model, observation_full, 0, config)

        grid = np.array(board_old).reshape(config.rows, config.columns)
        grid_next = step_andCounterStep(action, grid, opp_agent, mark, config, observation_full)
        done = check_if_done(grid_next)
        board_next = list(grid_next.flatten())
        reward, win = get_reward(done, action, board_old, board_next, mark)
        win_count += win

        # add to memory
        # extra_features = featureSpace(grid_next, config, mark)
        # extra_features = np.array([c['matrix'] for c in extra_features])
        # extra_features = np.array(extra_features).reshape(7, 4, 4, 1)
        # memory.add_to_memory(extra_features, action, reward)

        observation = board_next
        step_counter += 1

        if done[0]:
            print('done in %d steps' % step_counter, 'memory size:', len(memory.actions))
            # train after each game

            train_step(player_1_model, optimizer,
                       observations=np.array(memory.observations),
                       actions=np.array(memory.actions),
                       rewards=memory.rewards)

            break

# store model
checkpoint_path = "models/DQN/dqn_CNN_extra_4.ckpt"
player_1_model.save_weights(checkpoint_path)
# player_1_model.load_weights(checkpoint_path)

# model_path = "models/DQN/dqn_CNN_extra_3.h5"
# player_1_model.save(model_path)
# player_1_model = tf.keras.models.load_model(model_path)


# evaluate the agent
print('Eval against random')
get_win_percentages(player_1_agent, 'random', n_rounds=100)
print('\nEval against negamax')
get_win_percentages(player_1_agent, 'negamax', n_rounds=50)
