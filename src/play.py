import chess
import numpy as np
from pathlib import Path
from tensorflow import keras
from pgnparser import PGNParser

class Game:
    def __init__(self, model):
        self.model_path = str(Path(__file__).parents[1]) + "/model/" + model # path to data folder containing all games in pgn format
        self.model = self.initialize_model(self.model_path)
        self.board = chess.Board()
        self.parser = PGNParser(auto=False)

    def initialize_model(self, model_path):
        return keras.models.load_model(model_path)

    def predict_best_move(self, board):
        
        best_move = None
        val_of_best_move = 0

        for move in board.legal_moves:
            board.push(move)
            board_state = self.parser.convert_board_to_tensor(board, chess.BLACK).reshape(1, 6, 8, 8)

            val_of_curr_move = self.model.predict(board_state, verbose=0)

            if val_of_curr_move > val_of_best_move:
                val_of_best_move = val_of_curr_move
                best_move = move
            
            board.pop()
        
        return best_move
    
    def execute_move(self, move):
        self.board.push(move)
        print(self.board)
        print()
    
    def play(self):

        print(self.board)
        print()

        while not self.board.is_game_over():
            
            move = None

            while move is None:
                try:
                    inp = input("[WHITE] Play your move (format like 'b2b4'): ").lower().replace(" ", "")
                    move = chess.Move.from_uci(inp)

                    if not self.board.is_legal(move): 
                        print(f"\nThe move '{inp}' is not legal! Please try again...\n")
                        move = None

                except Exception:
                    print("\nInvalid move format (must be like 'b2b4')! Please try again...\n")
            
            self.execute_move(move)

            bot_move = self.predict_best_move(self.board)
            print(f"[BLACK] Pychessbot's move: '{bot_move.uci()}'\n")

            self.execute_move(bot_move)
        
        res = self.board.outcome()
        print(f"Game over! The result of the game is: {res.result()} (Winner: {res.winner})")

if __name__ == "__main__":
    game = Game("chess_model")
    game.play()
