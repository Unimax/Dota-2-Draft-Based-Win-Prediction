import enum
import chess
import numpy as np
import copy

from logging import getLogger

logger = getLogger(__name__)

Winner = enum.Enum("Winner", "BLACK WHITE DRAW")

pieces_order = 'KQRBNPkqrbnp'

pieces = {pieces_order[i]: i for i in range(12)}

print(pieces)

class MyChessEnv:
    def __init__(self):
        self.board = chess.Board()
        self.moves_count = 0
        self.winner = None  # type: Winner
        self.resigned = False
        self.score = None

    def reset(self):
        self.board = chess.Board()
        self.moves_count = 0
        self.winner = None
        self.resigned = False
        self.score = None
        return self

    def update(self, board):
        self.board = chess.Board(board)
        # TODO check if there is any winner and update the following
        self.winner = None
        self.resigned = False
        self.score = None
        return self

    def _resign(self):
        self.resigned = True
        if self.white_to_move:
            self.winner = Winner.black
            self.score = "0-1"
        else:
            self.winner = Winner.white
            self.score = "1-0"

    def step(self, action: str, check_over=True):
        """
        Takes an action and updates the game state
        :param str action: action to take in uci notation
        :param boolean check_over: whether to check if game is over
        """
        if check_over and action is None:
            self._resign()
            return

        self.board.push_uci(action)

        self.moves_count += 1

        if check_over and self.board.result(claim_draw=True) != "*":
            if self.winner is None:
                self.score = self.board.result(claim_draw=True)
                if self.score == '1-0':
                    self.winner = Winner.white
                elif self.score == '0-1':
                    self.winner = Winner.black
                else:
                    self.winner = Winner.draw
            else:
                print(" debug there is already a winner!")


    @property
    def done(self):
        return self.winner is not None

    @property
    def white_won(self):
        return self.winner == Winner.white

    @property
    def white_to_move(self):
        return self.board.turn == chess.WHITE

    def copy(self):
        env = copy.copy(self)
        env.board = copy.copy(self.board)
        return env

    def render(self):
        print("\n")
        print(self.board)
        print("\n")