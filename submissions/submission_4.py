"""
https://towardsdatascience.com/playing-connect-4-with-deep-q-learning-76271ed663ca

"""
import numpy as np
import tensorflow as tf
import random

# model_path = "models/DQN/dqn_CNN_extra_3.h5"
model_path = "dqn_CNN_extra_3.h5"
player_1_model = tf.keras.models.load_model(model_path)


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


def player_1_agent(observation, configuration):

    # predict action
    action, prob_weights = get_action(player_1_model, observation, 0, configuration)

    # check if valid
    if observation['board'][action] == 0:
        return action
    else:
        valid_moves = [col for col in range(configuration.columns) if observation.board[col] == 0]
        action = random.choice(valid_moves)
        return action
