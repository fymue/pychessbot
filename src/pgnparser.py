import chess
import chess.pgn
import os
import numpy as np
from pathlib import Path

class PGNParser:
    def __init__(self, auto=True, max_size=10000):
        self.max_size = max_size # maximum size of samples to be stored in X (training data)
        self.size = 0 # will be set after parge_pgn() gets called in parse_pgns()

        if auto:
            self.data_path = str(Path(__file__).parents[1]) + "/data/" # path to data folder containing all games in pgn format
            self.X, self.y = self.parse_pgns(self.data_path) # store every board state as well as the "goodness" of the state

    def parse_pgns(self, pgn_dir):
            # store the board states and their associated "goodness" values
            # for every pgn avaiable in the database
            
            # parse every pgn (containing multiple games) and store each game as well as the winner
            pgns = []
            for pgn_file in os.listdir(pgn_dir):
                if pgn_file.endswith(".pgn"):
                    pgn = self.parse_pgn(pgn_dir + pgn_file)
                    if pgn: pgns.append(pgn)

            X = np.empty((self.size, 7, 8, 8), dtype=np.int8)
            y = np.empty(self.size, dtype=np.uint8)

            i = 0 # counter for all board states added to X

            for fc, pgn in enumerate(pgns, 1):
                for gc, game in enumerate(pgn, 1):

                    if gc % 10 == 0: print(f"[File {fc}/{len(pgns)}] Parsing game {gc}/{len(pgn)} resulting in checkmate ({i} board state samples generated)\n")

                    game_board = game[0].board() # the game board (contains all moves played during the current game)
                    random_board = chess.Board() # a random board to execute random moves as a replacement for the loser's moves

                    winner = game[1]

                    moves_played = tuple(game[0].mainline_moves()) # get all moves played during the game

                    # figure out who won the game
                    if winner == chess.WHITE:
                        # if white won, all even moves by index (move 0, move 2, move 4 etc.) are "good" moves

                        good_move_start = 0
                        offset = 1 if len(moves_played) % 2 != 0 else 0
                    else:
                        # if black won, all uneven moves by index (move 1, move 3, move 5 etc.) are "good" moves
                        # since loop over moves below starts at the index of the first "good" move,
                        # and black won this game, white's opener has to be labeled as a "bad" move

                        good_move_start = 1
                        offset = 0 if len(moves_played) % 2 != 0 else 1

                        game_board.push(moves_played[0]) # play the 1st move since the loop starts at index 1 instead of 0

                        random_move = tuple(random_board.pseudo_legal_moves)[np.random.randint(0, random_board.pseudo_legal_moves.count())]
                        random_board.push(random_move)
                        bad_board_state = self.convert_board_to_tensor(random_board, winner)

                        X[i] = bad_board_state
                        y[i] = 0
                        i += 1

                    for move in range(good_move_start, len(moves_played) - offset, 2):
                        # convert board states to 6x8x8 tensor and label it correctly ("good"/"bad" move)
                        # for bad moves, make a random move from all possible moves of the current board state 
                        # (because all games are GrandMaster games and thus (early) moves are not necessarily bad)

                        # play the next move of the game and convert the resulting board state to a tensor
                        
                        game_board.push(moves_played[move])
                        good_board_state = self.convert_board_to_tensor(game_board, winner)

                        # set the random board (for the random, "bad" moves) to the current state of the game board,
                        # execute a random "bad" move and convert the resulting board state to a tensor 
                        random_board.set_fen(game_board.fen())
                        random_move = tuple(random_board.pseudo_legal_moves)[np.random.randint(0, random_board.pseudo_legal_moves.count())]
                        random_board.push(random_move)

                        bad_board_state = self.convert_board_to_tensor(random_board, winner)

                        # play the actual next move in the game so the actual next move is correct
                        game_board.push(moves_played[move+1]) 

                        X[i] = good_board_state
                        y[i] = 1 # "good" moves get a 1

                        X[i+1] = bad_board_state
                        y[i+1] = 0 # "bad" moves get a 0 

                        i += 2
                    
                    if offset:
                        # edge case: loop above increments by 2 every iteration
                        # -> in some cases the last move will not be executed in the loop
                        # -> has to be "manually" executed outside of the loop

                        move += 2
                        game_board.push(moves_played[move])

                        # the last move of a checkmate game is always the winning move
                        good_board_state = self.convert_board_to_tensor(game_board, winner)
                        X[i] = good_board_state
                        y[i] = 1

                        i += 1

            print(f"[File {fc}/{len(pgns)}] Finished! Parsed {gc}/{len(pgn)} games resulting in checkmate ({i} board state samples generated)\n")

            return X, y

    def parse_pgn(self, pgn_file):
        # read a pgn file containing multiple games
        # and find the winner of every game in order
        # to assign a "goodness" value for every move

        print(f"Reading games from PGN file '{pgn_file}' (this might take a few seconds)...")
        games = []

        try:
            pgn = open(pgn_file)
            game = chess.pgn.read_game(pgn) # read the first game

            while game != None and self.size < self.max_size:
                res = game.headers["Result"]

                """
                match res:
                    case "1-0": winner = chess.WHITE
                    case "0-1": winner = chess.BLACK
                    case _: winner = None
                """

                if res == "1-0": winner = chess.WHITE
                elif res == "0-1": winner = chess.BLACK
                else: winner = None
                
                total_moves = len(tuple(game.mainline_moves())) # total moves of this game

                if winner and total_moves >= 2: 
                    # store the current game and winner 
                    # (only if game ended in checkmate and game was "valid" 
                    # ( -> min. 2 moves; sometimes PGN contains falsely formatted games))

                    self.size +=  total_moves # add number of moves of current game to total size of X
                    games.append((game, winner))

                game = chess.pgn.read_game(pgn) # read the next game

            pgn.close()
        
        except Exception as e:
            print(f"Some game(s) from {pgn_file} could not be read!")
            print("There seems to be something wrong with the PGN format of the file.")
            print("Consider removing it from the data folder.")
            print(e)

        # only return games list if it contains at least 1 game (might not if self.max_size is already reached)
        return games if games else None
    

    def convert_board_to_tensor(self, board, winner):
        # convert the current board state to a 6x8x8 tensor (1 8x8 board for every figure)

        board_state = np.zeros((7, 8, 8), dtype=np.int8)
        piece_map = board.piece_map()

        white_val = 1 if winner == chess.WHITE else -1
        black_val = -white_val

        layer_indices = {"k" : 0, "p" : 1, "r" : 2, "b" : 3, "n" : 4, "q" : 5}

        for pos in piece_map:
            curr_piece = piece_map[pos].symbol()

            # calculate the correct index for the current tensor value
            row = np.abs(pos // 8 - 7)
            col = pos % 8
            layer = layer_indices[curr_piece.lower()]

            board_state[layer, row, col] = white_val if curr_piece.isupper() else black_val
        
        board_state[6, :, :] = board.turn * 1 # last column represents who's turn it is

        return board_state
    
    def convert_board_to_tensor2(self, board, winner):
        # different method of convert the current board state
        # (taken from https://github.com/geohot/twitchchess/blob/master/state.py)

        bstate = np.zeros(64, np.uint8)

        figs = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
                "p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}

        for i in range(64):
            pp = board.piece_at(i)
        if pp is not None:
            bstate[i] = figs[pp.symbol()]

        if board.has_queenside_castling_rights(chess.WHITE):
            bstate[0] = 7
        if board.has_kingside_castling_rights(chess.WHITE):
            bstate[7] = 7
        if board.has_queenside_castling_rights(chess.BLACK):
            bstate[56] = 8+7
        if board.has_kingside_castling_rights(chess.BLACK):
            bstate[63] = 8+7

        if board.ep_square is not None:
            bstate[board.ep_square] = 8

        bstate = bstate.reshape(8, 8)

        # binary state
        state = np.zeros((5, 8, 8), np.uint8)

        # 0-3 columns to binary
        state[0] = (bstate>>3)&1
        state[1] = (bstate>>2)&1
        state[2] = (bstate>>1)&1
        state[3] = (bstate>>0)&1

        # 4th column is who's turn it is
        state[4] = (board.turn * 1)

        # 257 bits according to readme
        return state

    def save_training_data(self, X, y):
        # save the training data to a .npz file
        # so it doesn't have to be recalculated every time

        np.savez_compressed(f"{self.data_path}training_data_{self.max_size}", X=X, y=y)

        

if __name__ == "__main__": 
    pgn_parser = PGNParser(max_size=1000000)
    pgn_parser.save_training_data(pgn_parser.X, pgn_parser.y)