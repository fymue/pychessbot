#!/usr/bin/env python3

import chess, chess.svg, flask, sunfish, argparse, os
import numpy as np
from pathlib import Path
from tensorflow import keras
from time import sleep
from pgnparser import PGNParser
from model import Model
from sys import argv

app = flask.Flask(__name__)
path = str(Path(__file__).parents[1])

class Game:
    def __init__(self, model, bot_move_delay=0):
        # load a model stored in model/, create an empty board to play on

        print(os.listdir())
        self.model_path =  path + "/model/"
        self.bot_move_delay = bot_move_delay
        self.board = None
        self.update_svg_board(None, "src/static/board.svg") # initialize/update the svg game board as empty
        self.model = self.initialize_model(self.model_path + model)
        self.parser = PGNParser(auto=False)

    def initialize_model(self, model_path):
        # load a previously trained model

        return keras.models.load_model(model_path)

    @staticmethod
    def update_svg_board(board, path):
        with open(path, "w") as fout: fout.write(chess.svg.board(board))

    def predict_best_move(self, board, model=None):
        # predict the best move from all possible moves
        # based on the current board state

        if model is None: model = self.model

        # get all legal moves from here (excluding moves that put the king in check)
        legal_moves = tuple(board.legal_moves)

        if not legal_moves:
            # if no moves that don't put the king in check are possible,
            # play one of those (means that the game is over)
            pseudo_legal_moves = board.pseudo_legal_moves
            legal_moves_uci = {move.uci() for move in legal_moves}
            pseudo_legal_moves_uci = {move.uci for move in pseudo_legal_moves}
            legal_moves = [chess.Move.from_uci(move) for move in legal_moves_uci ^ pseudo_legal_moves_uci]

        # calculate the board state tensor for every possible move
        possible_boards = np.empty((len(legal_moves), 8, 8, 6))

        for i, move in enumerate(legal_moves):
            board.push(move)
            possible_boards[i] = self.parser.convert_board_to_tensor(board, chess.BLACK)
            board.pop()
        

        # find the move that resulted in the biggest output value
        # and assume, that that move is the best one
        vals_of_boards = model.predict(possible_boards, verbose=0)
        best_move_i = np.argmax(vals_of_boards)
        best_move = legal_moves[best_move_i]
        val_of_best_move = vals_of_boards[best_move_i]
        
        return best_move
    
    def execute_move(self, move, quiet=True):
        # execute a move and print the updated board
        
        turn = self.board.turn # whose turn is it?
        is_pawn = self.board.piece_type_at(move.from_square) == chess.PAWN # is piece thats about to move a pawn?
        to_square = move.to_square // 8 # which row (1-8) does the piece move to?

        # check if pawn promotion is possible after move
        if (turn == chess.WHITE and is_pawn and  to_square == 7) or (turn == chess.BLACK and is_pawn and to_square == 0):
            move.promotion = chess.QUEEN

        self.board.push(move)

        if not quiet:
            print(self.board)
            print()
        
        self.update_svg_board(self.board, path + "/src/static/board.svg")

    
    def random_move(self, board):
        # play a random move
        # (needed e.g. for the bot playing against itself
        # so it doesn't play the same game every time)

        possible_moves = tuple(board.pseudo_legal_moves)
        return possible_moves[np.random.randint(0, len(possible_moves))]
    
    def get_game_result(self, board):
        res = board.outcome()

        if res.winner is None:
            winner = "Draw"
        elif res.winner == chess.WHITE:
            winner = "White"
        else:
            winner = "Black"

        if res:
            print(f"Game over! The result of the game is: {res.result()} (Winner: {winner} in {board.fullmove_number} moves)")
        else:
            print("The game was stopped due to it probably never coming to an end (over 50 moves played).")

        return

    def play_vs_player(self, quiet=False):
        # play a chess game against the bot
        self.board = chess.Board()

        if not quiet:
            print(self.board)
            print()

        while not self.board.is_game_over() and not self.board.is_fifty_moves():
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

        self.get_game_result(self.board)

        return

    def play_vs_self(self, quiet=False):
        # let the bot play a game against itself
        
        self.board = chess.Board()

        if not quiet:
            print(self.board)
            print()

        while not self.board.is_game_over() and not self.board.is_fifty_moves():
            # run the loop until the game is over (checkmate)
            
            # ca. 80% of the time, play the best move; ca. 20% of the time, play a random (bad) move
            bot_move = self.random_move(self.board) if np.random.random() <= 0.2 else self.predict_best_move(self.board)
            sleep(self.bot_move_delay)
            print(f"[WHITE] Pychessbot's move: '{bot_move.uci()}'\n")
            self.execute_move(bot_move, quiet=quiet)

            bot_move = self.random_move(self.board) if np.random.random() <= 0.2 else self.predict_best_move(self.board)
            sleep(self.bot_move_delay)
            print(f"[BLACK] Pychessbot's move: '{bot_move.uci()}'\n")
            self.execute_move(bot_move, quiet=quiet)
        
        self.get_game_result(self.board)

        return
    
    def play_vs_model(self, opp_model, main_model=None, quiet=False):
        # let two models play against each other
        
        self.board = chess.Board()

        if main_model is None: 
            main_model = self.model
        else:
            main_model = self.initialize_model(self.model_path + main_model)

        opp_model = self.initialize_model(self.model_path + opp_model)

        if not quiet:
            print(self.board)
            print()

        while not self.board.is_game_over() and not self.board.is_fifty_moves():
            # run the loop until the game is over (checkmate)
            
            if self.board.fullmove_number == 1: bot_move = self.random_move(self.board)
            else: bot_move = self.predict_best_move(self.board, model=main_model)

            sleep(self.bot_move_delay)
            print(f"[WHITE] Pychessbot's move (main model): '{bot_move.uci()}'\n")
            self.execute_move(bot_move, quiet=quiet)

            bot_move = self.predict_best_move(self.board, model=opp_model)
            sleep(self.bot_move_delay)
            print(f"[BLACK] Pychessbot's move (opp model): '{bot_move.uci()}'\n")
            self.execute_move(bot_move, quiet=quiet)
        
        self.get_game_result(self.board)

        return
    
    def play_vs_sunfish(self, quiet=False):
        # play against the sunfish chess engine
        # (https://github.com/thomasahle/sunfish/)

        self.board = chess.Board()
        sunfish_board = sunfish.Position(sunfish.initial, 0, (True,True), (True,True), 0, 0)
        sunfish_searcher = sunfish.Searcher()

        if not quiet:
            print(self.board)
            print()

        move_c = 0

        while not self.board.is_game_over() and not self.board.is_fifty_moves():
            # run the loop until the game is over (checkmate)

            bot_move = self.predict_best_move(self.board) if self.board.fullmove_number > 1 else self.random_move(self.board)
            bot_move_uci = bot_move.uci()

            sleep(self.bot_move_delay)
            print(f"[WHITE] Pychessbot's move: '{bot_move_uci}'\n")
            self.execute_move(bot_move, quiet=quiet)

            start_sq, end_sq = bot_move_uci[:2], bot_move_uci[2:] # start square and end square of last pychessbot move
            bot_move_to_sunfish = (sunfish.parse(start_sq), sunfish.parse(end_sq)) # convert to sunfish's move format
            sunfish_board = sunfish_board.move(bot_move_to_sunfish) # execute move on sunfish board

            # let sunfish make its move
            sunfish_move, sunfish_score = sunfish_searcher.search(sunfish_board, secs=1) # sunfish best move prediction
            sunfish_board = sunfish_board.move(sunfish_move) # play predicted move on sunfish board

            # adjust move to match turn (black) and convert sunfish move to python-chess move
            sunfish_move_uci = sunfish.render(119-sunfish_move[0]) + sunfish.render(119-sunfish_move[1])
            
            print(f"[BLACK] Sunfish's move: '{sunfish_move_uci}'\n")
            self.execute_move(chess.Move.from_uci(sunfish_move_uci), quiet=quiet) # play sunfish's move on main board
        
        self.get_game_result(self.board)
        
        return

@app.route("/")
def init_page(): return flask.render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def start_game():
    select = str(flask.request.form.get("gamemode"))

    game = Game("chess_model", bot_move_delay=1)

    if select == "sunfish": game.play_vs_sunfish(quiet=True)
    elif select == "self": game.play_vs_self(quiet=True)
    elif select == "model": game.play_vs_model(quiet=True)
    else: game.play_vs_self(quiet=True)

    return "OK"
    


if __name__ == "__main__":
    
    # parse the command line arguments
    # (if no argument -> launch GUI as webpage on localhost:5000)
    # (else: play/watch a game on the command line)

    passed_mode_args = len([True for el in argv if el[0] == "-"])

    if len(argv) == 1: 
        app.run()
    elif passed_mode_args > 1:
        print(f"Please provide at most 1 game mode option (found {passed_mode_args})!")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--player", "-p", action="store_true", help="Start a new game between PyChessBot vs. a player (you)")
        parser.add_argument("--self", "-s", action="store_true", help="Let PyChessBot play a game against itself")
        parser.add_argument("--sunfish", "-sf", action="store_true", help="Let PyChessBot play a game against the Sunfish engine")
        parser.add_argument("--model", "-m", nargs=2, metavar=("model1", "model2"), type=str, help="Let two models from pychessbot/model/ play against each other")

        args = parser.parse_args()

        game = Game("chess_model")

        if args.player: game.play_vs_player()
        elif args.self: game.play_vs_self()
        elif args.model: game.play_vs_model(args.model[1], args.model[0])
        elif args.sunfish: game.play_vs_sunfish()

    Game.update_svg_board(None, path + "/src/static/board.svg")

        

