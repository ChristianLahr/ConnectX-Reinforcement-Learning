def my_agent(obs, config):
    """
    Name: agent_longestConnections_blocker
    choose the column, with which the longest row of your coins is possible """
    import random
    import numpy as np


    def coins_in_a_row(window, obs):
        # l = np.where(np.array(window) == (obs.mark % 2) + 1)[0]
        n = 0
        for value in window:
            if value == obs.mark:
                n += 1
            else:
                return n
        return n


    def featureSpace(grid, obs, config):
        feature = []
        for col in range(config.columns):
            feature.append({})
            feature_per_pos = np.zeros([4, 4])  # (left/up, right/down, overall, win) x (horizontal, diagonal_down, vertical, diagonal_up)

            n_coins_in_col = ((grid[:, 6] == 1) | (grid[:, 6] == 2)).sum()
            if n_coins_in_col == config.rows:  # row is already full
                feature[col]['full'] = 1
            else:
                feature[col]['full'] = 0
                row = n_coins_in_col  # new empty row in this column

                ### horizontal
                # left
                l2 = list(range(max(min(col - 1, config.columns), -1), max(min(col - 1 - config.inarow, config.columns), -1), -1))
                l1 = [row] * len(l2)
                n = min(len(l1), len(l2))
                feature_per_pos[0, 0] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], obs)

                # right
                l2 = list(range(max(min(col + 1, config.columns), -1), max(min(col + 1 + config.inarow, config.columns), -1)))
                l1 = [row] * len(l2)
                n = min(len(l1), len(l2))
                feature_per_pos[1, 0] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], obs)

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
                feature_per_pos[1, 2] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], obs)

                # overall
                feature_per_pos[2, 2] = feature_per_pos[0, 2] + feature_per_pos[1, 2]
                # win move
                feature_per_pos[3, 2] = int(feature_per_pos[2, 2] + 1 >= config.inarow)

                ### diagonal down
                # left (links oben)
                l1 = list(range(max(min(row + 1, config.rows), -1), max(min(row + 1 + config.inarow, config.rows), -1)))
                l2 = list(range(max(min(col - 1, config.columns), -1), max(min(col - 1 - config.inarow, config.columns), -1), -1))
                n = min(len(l1), len(l2))
                feature_per_pos[0, 1] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], obs)

                # right (rechts unten)
                l1 = list(range(max(min(row - 1, config.rows), -1), max(min(row - 1 - config.inarow, config.rows), -1), -1))
                l2 = list(range(max(min(col + 1, config.columns), -1), max(min(col + 1 + config.inarow, config.columns), -1)))
                n = min(len(l1), len(l2))
                feature_per_pos[1, 1] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], obs)

                # overall
                feature_per_pos[2, 1] = feature_per_pos[0, 1] + feature_per_pos[1, 1]
                # win move
                feature_per_pos[3, 1] = int(feature_per_pos[2, 1] + 1 >= config.inarow)

                ### diagonal up
                # left (links unten)
                l1 = list(range(max(min(row - 1, config.rows), -1), max(min(row - 1 - config.inarow, config.rows), -1), -1))
                l2 = list(range(max(min(col - 1, config.columns), -1), max(min(col - 1 - config.inarow, config.columns), -1), -1))
                n = min(len(l1), len(l2))
                feature_per_pos[0, 3] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], obs)

                # right (rechts oben)
                l1 = list(range(max(min(row + 1, config.rows), -1), max(min(row + 1 + config.inarow, config.rows), -1)))
                l2 = list(range(max(min(col + 1, config.columns), -1), max(min(col + 1 + config.inarow, config.columns), -1)))
                n = min(len(l1), len(l2))
                feature_per_pos[1, 3] = coins_in_a_row(np.flip(grid, 0)[l1[:n], l2[:n]], obs)

                # overall
                feature_per_pos[2, 3] = feature_per_pos[0, 3] + feature_per_pos[1, 3]
                # win move
                feature_per_pos[3, 3] = int(feature_per_pos[2, 1] + 1 >= config.inarow)

                feature[col]['matrix'] = feature_per_pos
                feature[col]['maxRow'] = feature_per_pos[2, :].max()
                feature[col]['win'] = feature_per_pos[3, :].sum() > 0

        return feature


    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, piece, config):
        next_grid = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = piece
        return next_grid


    # Returns True if dropping piece in column results in game win
    def check_winning_move(obs, config, col, piece):
        import numpy as np

        # Convert the board to a 2D grid
        grid = np.asarray(obs.board).reshape(config.rows, config.columns)
        next_grid = drop_piece(grid, col, piece, config)
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(next_grid[row, col:col + config.inarow])
                if window.count(piece) == config.inarow:
                    return True
        # vertical
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(next_grid[row:row + config.inarow, col])
                if window.count(piece) == config.inarow:
                    return True
        # positive diagonal
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(next_grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if window.count(piece) == config.inarow:
                    return True
        # negative diagonal
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(next_grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if window.count(piece) == config.inarow:
                    return True
        return False


    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]

    # If we have a winning move, take it
    for col in valid_moves:
        if check_winning_move(obs, config, col, obs.mark):
            return col

            # Else if the opponent has a winning move, block it
    opponent = (obs.mark % 2) + 1
    for col in valid_moves:
        if check_winning_move(obs, config, col, opponent):
            return col

    # Else try to build a longer connection of my coins
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    f = featureSpace(grid, obs, config)

    # choose column where biggest nuber of coins in a row is possible
    next_move = valid_moves[np.argmax([f[col]['maxRow'] for col in valid_moves])]

    return next_move  # random.choice(valid_moves)



if __name__ == '__main__':
    from kaggle_environments import make
    env = make("connectx", debug=True)
    env.run([my_agent, my_agent])