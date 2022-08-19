import chess
import numpy as np
from pathlib import Path
from tensorflow import keras
from time import sleep
from pgnparser import PGNParser

class Game:
    def __init__(self, model, bot_move_delay=0):
        # load a model stored in model/, create an empty board to play on

        self.model_path = str(Path(__file__).parents[1]) + "/model/" + model
        self.bot_move_delay = bot_move_delay
        self.model = self.initialize_model(self.model_path)
        self.board = None
        self.parser = PGNParser(auto=False)

    def initialize_model(self, model_path):
        # load a previously trained model

        return keras.models.load_model(model_path)

    def predict_best_move(self, board):
        # predict the best move from all possible moves
        # based on the current board state

        #possible_boards = np.empty((board.legal_moves.count(), 6, 8, 8))
        possible_boards = np.empty((board.legal_moves.count(), 7, 8, 8))

        legal_moves = tuple(board.legal_moves)

        # calculate the board state tensor for every possible move
        for i, move in enumerate(legal_moves):
            board.push(move)
            possible_boards[i] = self.parser.convert_board_to_tensor(board, chess.BLACK)
            board.pop()

        # find the move that resulted in the biggest output value
        # and assume, that that move is the best one
        vals_of_boards = self.model.predict(possible_boards, verbose=0)
        best_move_i = np.argmax(vals_of_boards)
        best_move = legal_moves[best_move_i]
        val_of_best_move = vals_of_boards[best_move_i]
        print(vals_of_boards)
        print(legal_moves)
        
        return best_move
    
    def execute_move(self, move, quiet=True):
        # execute a move and print the updated board

        self.board.push(move)

        if not quiet:
            print(self.board)
            print()
    
    def random_move(self, board):
        # play a random move
        # (needed e.g. for the bot playing against itself
        # so it doesn't play the same game every time)

        possible_moves = tuple(board.pseudo_legal_moves)
        return possible_moves[np.random.randint(0, len(possible_moves))]
    
    def play_vs_player(self, quiet=False):
        # run the game loop

        self.board = chess.Board()

        if not quiet:
            print(self.board)
            print()

        while not self.board.is_game_over():
            # run the loop until the game is over (checkmate)
            
            move = None

            while move is None:
                try:
                    # parse the user's move
                    inp = input("[WHITE] Play your move (format like 'b2b4'): ").lower().replace(" ", "")
                    move = chess.Move.from_uci(inp)

                    if not self.board.is_legal(move): 
                        print(f"\nThe move '{inp}' is not legal! Please try again...\n")
                        move = None

                except Exception:
                    print("\nInvalid move format (must be like 'b2b4')! Please try again...\n")
            
            self.execute_move(move, quiet=quiet)

            # let the model predict the best move
            bot_move = self.predict_best_move(self.board)
            sleep(self.bot_move_delay)
            print(f"[BLACK] Pychessbot's move: '{bot_move.uci()}'\n")

            self.execute_move(bot_move, quiet=quiet)
        
        res = self.board.outcome()
        print(f"Game over! The result of the game is: {res.result()} (Winner: {res.winner})")

    def play_vs_self(self, quiet=False):
        
        self.board = chess.Board()

        if not quiet:
            print(self.board)
            print()

        move_c = 0

        while not self.board.is_game_over() and move_c < 100:
            # run the loop until the game is over (checkmate)
            
            if move_c == 0: bot_move = self.random_move(self.board)
            else: bot_move = self.predict_best_move(self.board)

            sleep(self.bot_move_delay)
            print(f"[WHITE] Pychessbot's move: '{bot_move.uci()}'\n")
            self.execute_move(bot_move, quiet=quiet)

            bot_move = self.predict_best_move(self.board)
            sleep(self.bot_move_delay)
            print(f"[BLACK] Pychessbot's move: '{bot_move.uci()}'\n")
            self.execute_move(bot_move, quiet=quiet)
        
            move_c += 1
        
        res = self.board.outcome()
        print(f"Game over! The result of the game is: {res.result()} (Winner: {res.winner} in {move_c} moves)")

        return res.winner


if __name__ == "__main__":
    game = Game("chess_model_v2")
    game.play_vs_self()
