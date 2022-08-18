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
        self.board = chess.Board()
        self.parser = PGNParser(auto=False)

    def initialize_model(self, model_path):
        # load a previously trained model

        return keras.models.load_model(model_path)

    def predict_best_move(self, board):
        # predict the best move from all possible moves
        # based on the current board state

        possible_boards = np.empty((board.legal_moves.count(), 6, 8, 8))
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
        
        return best_move
    
    def execute_move(self, move):
        # execute a move and print the updated board

        self.board.push(move)
        print(self.board)
        print()
    
    def play(self):
        # run the game loop

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
            
            self.execute_move(move)

            # let the model predict the best move
            bot_move = self.predict_best_move(self.board)
            sleep(self.bot_move_delay)
            print(f"[BLACK] Pychessbot's move: '{bot_move.uci()}'\n")

            self.execute_move(bot_move)
        
        res = self.board.outcome()
        print(f"Game over! The result of the game is: {res.result()} (Winner: {res.winner})")

if __name__ == "__main__":
    game = Game("chess_model")
    game.play()
