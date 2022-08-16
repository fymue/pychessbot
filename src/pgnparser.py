import chess
import chess.pgn
import os
import numpy as np
from pathlib import Path

class PGNParser:
    def __init__(self, max_size=10000):
        self.max_size = max_size # maximum size of X (training data)
        self.data_path = str(Path(__file__).parents[1]) + "/data/" # path to data folder containing all games in pgn format
        self.size = 0 # will be set after parge_pgn() gets called in parse_pgns()
        self.X, self.y = self.parse_pgns(self.data_path) # store every board state as well as the "goodness" of the state

    def parse_pgns(self, pgn_dir):
            # store the board states and their associated "goodness" values
            # for every pgn avaiable in the database
            
            # parse every pgn (containing multiple games) and store each game as well as the winner
            pgns = [self.parse_pgn(pgn_dir + pgn_file) for pgn_file in os.listdir(pgn_dir) if pgn_file.endswith(".pgn")]

            X = np.empty((self.size, 8, 8, 6), dtype=np.int8)
            y = np.empty(self.size, dtype=np.uint8)

            i = 0 # counter for all board states added to X

            for fc, pgn in enumerate(pgns, 1):
                for gc, game in enumerate(pgn, 1):

                    if gc % 10 == 0: print(f"[File {fc}/{len(pgns)}] Parsing game {gc}/{len(pgn)} resulting in checkmate...\n")

                    game_board = game[0].board()
                    random_board = chess.Board()

                    winner = game[1]

                    moves_played = tuple(game[0].mainline_moves()) # get all moves played during the game

                    # figure out who won the game
                    if winner == chess.WHITE:
                        # if white won, all even moves by index (move 0, move 2, move 4 etc. are "good" moves)

                        good_move_start = 0
                        offset = 1 if len(moves_played) % 2 != 0 else 0
                    else:
                        # if black won, all uneven moves by index (move 1, move 3, move 5 etc. are "good" moves)
                        # since loop over moves below starts at the index of the first "good" move,
                        # and black won this game, white's opener has to be labeled as a "bad" move

                        good_move_start = 1
                        offset = 0 if len(moves_played) % 2 != 0 else 1
                        random_move = tuple(random_board.legal_moves)[np.random.randint(0, random_board.legal_moves.count())]
                        random_board.push(random_move)
                        bad_board_state = self.convert_board_to_tensor(random_board)
                        X[i] = good_board_state
                        y[i] = 0
                        i += 1

                    for move in range(good_move_start, len(moves_played) - offset, 2):
                        # convert board states to 8x8x6 tensor and label it correctly ("good"/"bad" move)
                        # for bad moves, make a random move from all possible moves of the current board state 
                        # (because all games are GrandMaster games and thus (early) moves are not necessarily bad)

                        game_board.push(moves_played[move])
                        good_board_state = self.convert_board_to_tensor(game_board)

                        random_board.set_fen(game_board.fen())
                        random_move = tuple(random_board.legal_moves)[np.random.randint(0, random_board.legal_moves.count())]
                        random_board.push(random_move)
                        bad_board_state = self.convert_board_to_tensor(random_board)

                        game_board.push(moves_played[move+1])

                        X[i] = good_board_state
                        y[i] = 1
                        X[i+1] = bad_board_state
                        y[i] = 0
                        i += 2
                    
                    if offset:
                        # edge case: loop above increments by 2 every iteration
                        # in some cases the last move will not be executed in the loop

                        move += 2
                        game_board.push(moves_played[move])
                        good_board_state = self.convert_board_to_tensor(game_board)

                        X[i] = good_board_state
                        y[i] = 1
                        i += 1
        
            return X, y

    def parse_pgn(self, pgn_file):
        # read a pgn file containing multiple games
        # and find the winner of the path in order
        # to assign a "goodness" value for every move

        games = []

        pgn = open(pgn_file)
        game = chess.pgn.read_game(pgn)

        while game != None and self.size <= self.max_size:
            res = game.headers["Result"]

            match res:
                case "1-0": winner = chess.WHITE
                case "0-1": winner = chess.BLACK
                case _: winner = None
            
            if winner: 
                self.size += len(tuple(game.mainline_moves())) # add number of moves of current game to total size of X
                games.append((game, winner))

            game = chess.pgn.read_game(pgn)

        pgn.close()

        return games
    
    def convert_board_to_tensor(self, state):
        # convert the current board state to a
        # 8x8x6 tensor (1 8x8 board for every figure)

        board_state = np.zeros((8, 8, 6), dtype=np.int8)

        return board_state

    def save_training_data(self, X, y):
        # save the training data to a .npz file
        # so it doesn't have to be recalculated every time

        np.savez(self.data_path + "training_data", X=X, y=y)



if __name__ == "__main__": 
    pgn_parser = PGNParser(max_size=1000)
    pgn_parser.save_training_data(pgn_parser.X, pgn_parser.y)

    
