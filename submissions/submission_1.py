def my_agent(obs, config):

    import random
    import numpy as np

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
    for col in valid_moves:
        if check_winning_move(obs, config, col, obs.mark):
            return col
    opponent = (obs.mark%2)+1
    for col in valid_moves:
        if check_winning_move(obs, config, col, opponent):
            return col
    return random.choice(valid_moves)




if __name__ == '__main__':
    from kaggle_environments import make
    env = make("connectx", debug=True)
    env.run([my_agent, my_agent])