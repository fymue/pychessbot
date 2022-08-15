import chess
import chess.pgn
import os
import numpy as np
from pathlib import Path

class PGNParser:
    def __init__(self):
        self.pgns_path = str(Path(__file__).parents[1]) + "/data/" # path to data folder containing all games in pgn format
        self.board_states = self.parse_pgns(self.pgns_path) # store every board state as well as the "goodness" of the state

    def parse_pgns(self, pgn_dir):
            # store the board states and their associated "goodness" values
            # for every pgn avaiable in the database
            board_states = []
            pgns = [self.parse_pgn(pgn_dir + pgn_file) for pgn_file in os.listdir(pgn_dir)]
            board_states.extend([self.parse_game(game) for pgn in pgns for game in pgn])

            return board_states
    
    def parse_game(self, game):
        # parse a chess game
        # TODO: go over all moves, convert to tensor, label all winning moves as 1, all loser moves random move instead (label 0 or -1)

        board = game[0].board()
        winner = game[1]
        moves_played = game.mainline_moves()

        X = []
        y = []

        for move in moves_played:
            board.push(move)
            board_state = self.convert_board_to_tensor(board)

    def parse_pgn(self, pgn_file):
        # read a pgn file containing multiple games
        # and find the winner of the path in order
        # to assign a "goodness" value for every move

        games = []

        pgn = open(pgn_file)
        game = chess.pgn.read_game(pgn)

        while game != None:
            res = game.headers["Result"]

            match res:
                case "1-0": winner = chess.WHITE
                case "0-1": winner = chess.BLACK
                case _: winner = None
            
            if winner: games.append(game, winner)

            game = chess.pgn.read_game(pgn)

        return games
    
    def convert_board_to_tensor(self, state):
        # convert the current board state to a
        # 8x8x6 tensor (1 8x8 board for every figure)
        # board state is X, winner (state[1]) is y

        board_state = np.zeros((8, 8, 6), dtype=np.int8)

        return board_state

if __name__ == "__main__": PGNParser()
