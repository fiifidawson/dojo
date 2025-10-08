import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

class State:
    def __init__(self):
        # the board is represented by an n * n array
        # 1 represents a chessman of the player who moves first,
        # 0 represents an empty position
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_value = None
        self.end = None

    # compute the hash value for one state, it's unique
    def hash(self):
        if self.hash_value is None:
            self.hash_value = 0
            for i in np.nditer(self.data):
                self.hash_value = self.hash_value * 3 + 1
        return self.hash_value