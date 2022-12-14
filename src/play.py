#!/usr/bin/env python3

import chess, chess.svg, flask, sunfish, argparse
import numpy as np
from pathlib import Path
from tensorflow import keras
from time import sleep
from pgnparser import PGNParser
from model import Model
from sys import argv

game = None

move_history = ""

path = Path(__file__).absolute().parent.parent

app = flask.Flask(__name__)

class Game:

    depth = 0

    def __init__(self, model, bot_move_delay=0):
        # load a model stored in model/, create an empty board to play on

        self.model_path = path.joinpath("model").as_posix() + "/"
        self.bot_move_delay = bot_move_delay
        self.board = None
        self.move_c = 1
        self.update_move_history(None, None, None) # reset the move history before the start of a new game
        self.update_svg_board(None) # initialize/update the svg game board as empty
        self.model = self.initialize_model(self.model_path + model)

    def initialize_model(self, model_path):
        # load a previously trained model

        return keras.models.load_model(model_path)

    @staticmethod
    def update_svg_board(board):
        with open(path.joinpath("src/static/board.svg"), "w") as fout: fout.write(chess.svg.board(board))
    
    @staticmethod
    def evaluate_board_state(board, model, color):
        # calculate model output for a board state/position

        board_state = PGNParser.convert_board_to_tensor(board, color)
        board_state = board_state.reshape((1,) + board_state.shape)
        return model.predict(board_state, verbose=0)

    @staticmethod
    def alpha_beta(depth, board, model, color, alpha, beta, maximizing_player, n=5):
        # alpha beta pruning algorithm (determines best move to play)

        if depth == 0: return Game.evaluate_board_state(board, model, color)
        
        moves_to_check = Game.calc_move_scores(board, model, color, n) # only pick best n moves to further evaluate (to save time)

        if maximizing_player:
            val = np.NINF

            for move in moves_to_check:
                board.push(move)
                val = max(val, Game.alpha_beta(depth-1, board, model, color, alpha, beta, False))
                board.pop()

                alpha = max(alpha, val)

                if val >= beta: break
            
            return val
        
        else:
            val = np.Inf

            for move in moves_to_check:
                board.push(move)
                val = min(val, Game.alpha_beta(depth-1, board, model, color, alpha, beta, True))
                board.pop()

                beta = min(beta, val)

                if val <= alpha: break

            return val

    @staticmethod
    def calc_move_scores(board, model, color, n=1):
        # calculate the scores of all possible moves
        # and return the best n moves (sorted by score)

        legal_moves = np.array(tuple(board.legal_moves))

        if legal_moves.size == 0:
            # if no moves that don't put the king in check are possible,
            # play one of those (means that the game is over)

            pseudo_legal_moves = board.pseudo_legal_moves
            legal_moves_uci = {move.uci() for move in legal_moves}
            pseudo_legal_moves_uci = {move.uci() for move in pseudo_legal_moves}
            legal_moves = np.array([chess.Move.from_uci(move) for move in legal_moves_uci ^ pseudo_legal_moves_uci])

        possible_boards = np.empty((len(legal_moves), 8, 8, 6))

        for i, move in enumerate(legal_moves):
            board.push(move)
            possible_boards[i] = PGNParser.convert_board_to_tensor(board, color)
            board.pop()
        

        # find the move that resulted in the biggest output value
        # and assume, that that move is the best one
        vals_of_moves = model.predict(possible_boards, verbose=0).flatten()
        best_moves_i = np.argsort(vals_of_moves) # sort scores by index ascending

        best_n_moves = legal_moves[best_moves_i[best_moves_i.size - n:]]

        return best_n_moves

    @staticmethod
    def predict_best_move(board, model, color):
        # predict the best move from all possible moves
        # based on the current board state
        # using additional alpha-beta-pruning if depth is bigger 0

        best_5_moves = Game.calc_move_scores(board, model, color, n=5) # calculate 5 best moves based on model output
        best_move = best_5_moves[-1]

        if Game.depth > 0:
            # if user entered depth bigger than 0,
            # run additional alpha-beta-pruning to
            # search the game tree for a better move
            # until the max depth is reached

            best_move_val = np.NINF

            # only search the game tree starting with the 5 best moves
            for curr_move in best_5_moves:

                board.push(curr_move)
                curr_move_val = Game.alpha_beta(Game.depth, board, model, color, np.NINF, np.Inf, True)
                board.pop()

                if curr_move_val > best_move_val: best_move = curr_move

        return best_move

    @staticmethod
    def execute_move(move, board, move_c, quiet=True):
        # execute a move and print the updated board
        global move_history
        
        turn = board.turn # whose turn is it?
        is_pawn = board.piece_type_at(move.from_square) == chess.PAWN # is piece thats about to move a pawn?
        to_square = move.to_square // 8 # which row (1-8) does the piece move to?

        # check if pawn promotion is possible after move
        if (turn == chess.WHITE and is_pawn and  to_square == 7) or (turn == chess.BLACK and is_pawn and to_square == 0):
            move.promotion = chess.QUEEN

        board.push(move)

        Game.update_move_history(move, move_c, turn)

        if not quiet:
            print(board)
            print()
        
        Game.update_svg_board(board)

        return move_c + 1
    
    def random_move(self, board):
        # play a random move
        # (needed e.g. for the bot playing against itself
        # so it doesn't play the same game every time)

        possible_moves = tuple(board.pseudo_legal_moves)
        return possible_moves[np.random.randint(0, len(possible_moves))]
    
    @staticmethod
    def get_game_result(board):
        global move_history

        if game.board.is_game_over() or game.board.is_fifty_moves():
            res = board.outcome()

            if res.winner is None:
                winner = "Draw"
            elif res.winner == chess.WHITE:
                winner = "White"
            else:
                winner = "Black"

            if res:
                res_msg = f"Game over! The result of the game is: {res.result()} (Winner: {winner} in {board.fullmove_number} moves)"
            else:
                res_msg = "The game was stopped due to it probably never coming to an end (over 50 moves played)."
        
        else: 
            res_msg = ""

        with open(path.joinpath("src/static/move_history.txt"), "a") as fout:
            fout.write(res_msg)

        return res_msg
    
    @staticmethod
    def update_move_history(move, move_c, turn):
        # update the move history log file (is being read every second in js and displayed in iframe)

        file_path = path.joinpath("src/static/move_history.txt")

        if move is None: open(file_path, "w").close() # delete the move history if a game ends
        else:
            with open(file_path, "a") as fout:
                fout.write(f'{move_c}. [{"WHITE" if turn == chess.WHITE else "BLACK"}] {move.uci()}' + '<br>')

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
            
            self.move_c = self.execute_move(move, self.board, self.move_c, quiet=quiet)

            # let the model predict the best move
            bot_move = self.predict_best_move(self.board, self.model, chess.BLACK)

            sleep(self.bot_move_delay)
            print(f"{self.move_c + 1}. [BLACK] Pychessbot's move: '{bot_move.uci()}'\n")

            self.move_c = self.execute_move(bot_move, self.board, self.move_c,quiet=quiet)

        print(self.get_game_result(self.board))

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
            bot_move = self.random_move(self.board) if np.random.random() <= 0.2 else self.predict_best_move(self.board, self.model, chess.WHITE)
            sleep(self.bot_move_delay)
            print(f"{self.move_c + 1}. [WHITE] Pychessbot's move: '{bot_move.uci()}'\n")
            self.move_c = self.execute_move(bot_move, self.board, self.move_c, quiet=quiet)

            bot_move = self.random_move(self.board) if np.random.random() <= 0.2 else self.predict_best_move(self.board, self.model, chess.WHITE)
            sleep(self.bot_move_delay)
            print(f"{self.move_c + 1}. [BLACK] Pychessbot's move: '{bot_move.uci()}'\n")
            self.move_c = self.execute_move(bot_move, self.board, self.move_c, quiet=quiet)
        
        print(self.get_game_result(self.board))

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
            else: bot_move = self.predict_best_move(self.board, main_model, chess.WHITE)

            sleep(self.bot_move_delay)
            print(f"{self.move_c + 1}. [WHITE] Pychessbot's move (main model): '{bot_move.uci()}'\n")
            self.move_c = self.execute_move(bot_move, self.board, self.move_c, quiet=quiet)

            bot_move = self.predict_best_move(self.board, opp_model, chess.BLACK)
            sleep(self.bot_move_delay)
            print(f"{self.move_c + 1}. [BLACK] Pychessbot's move (opp model): '{bot_move.uci()}'\n")
            self.move_c = self.execute_move(bot_move, self.board, self.move_c, quiet=quiet)
        
        print(self.get_game_result(self.board))

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

        while not self.board.is_game_over() and not self.board.is_fifty_moves():
            # run the loop until the game is over (checkmate)

            bot_move = self.predict_best_move(self.board, self.model, chess.WHITE) if self.board.fullmove_number > 1 else self.random_move(self.board)
            bot_move_uci = bot_move.uci()

            sleep(self.bot_move_delay)
            print(f"{self.move_c + 1}. [WHITE] Pychessbot's move: '{bot_move_uci}'\n")
            self.move_c = self.execute_move(bot_move, self.board, self.move_c, quiet=quiet)

            start_sq, end_sq = bot_move_uci[:2], bot_move_uci[2:] # start square and end square of last pychessbot move
            bot_move_to_sunfish = (sunfish.parse(start_sq), sunfish.parse(end_sq)) # convert to sunfish's move format
            sunfish_board = sunfish_board.move(bot_move_to_sunfish) # execute move on sunfish board

            # let sunfish make its move
            sunfish_move, sunfish_score = sunfish_searcher.search(sunfish_board, secs=1) # sunfish best move prediction
            sunfish_board = sunfish_board.move(sunfish_move) # play predicted move on sunfish board

            # adjust move to match turn (black) and convert sunfish move to python-chess move
            sunfish_move_uci = sunfish.render(119-sunfish_move[0]) + sunfish.render(119-sunfish_move[1])
            
            print(f"{self.move_c + 1}. [BLACK] Sunfish's move: '{sunfish_move_uci}'\n")
            self.move_c = self.execute_move(chess.Move.from_uci(sunfish_move_uci), self.board, self.move_c, quiet=quiet) # play sunfish's move on main board
        
        print(self.get_game_result(self.board))
        
        return

@app.route("/")
def init_page(): return flask.render_template("index.html")

@app.route("/", methods=["GET", "POST"])
def run_game():
    global game, move_history

    select = str(flask.request.form.get("gamemode"))

    if not game and select == "sunfish": 
        game = Game("chess_model_v2", bot_move_delay=1)
        game.play_vs_sunfish(quiet=True)

    elif not game and select == "self": 
        game = Game("chess_model_v2", bot_move_delay=1)
        game.play_vs_self(quiet=True)

    elif not game and select == "player":
        # start a game between a (human) player and PyChessBot
        game = Game("chess_model_v2", bot_move_delay=1)
        game.board = chess.Board()
        Game.update_svg_board(game.board)

    elif flask.request.form.get("reset"):
        move_history = ""
        game = None
        Game.update_svg_board(None)
        Game.update_move_history(None, None, None)

    else:
        move = None
        if not game: game = Game("chess_model_v2", bot_move_delay=1)

        inp = str(flask.request.form.get("enteredMove"))

        while move is None:
            try:
                # parse the user's move
                move = chess.Move.from_uci(inp)

                if not game.board.is_legal(move): 
                    move = None
                    return f"The move '{inp}' is not legal! Please try again..."

            except Exception:
                return f"Invalid move format '{inp}' (must be like 'b2b4')! Please try again..."

        game.move_c = Game.execute_move(move, game.board, game.move_c, quiet=True)
        move_history += f'[{"WHITE" if game.board.turn == chess.WHITE else "BLACK"}] {move.uci()}' + '<br>'

        if game.board.is_game_over() or game.board.is_fifty_moves():  return Game.get_game_result(game.board)

        bot_move = Game.predict_best_move(game.board, game.model, chess.BLACK)

        sleep(1)
        print(f"[BLACK] Pychessbot's move: '{bot_move.uci()}'\n")

        game.move_c = Game.execute_move(bot_move, game.board, game.move_c, quiet=True)
        move_history += f'[{"WHITE" if game.board.turn == chess.WHITE else "BLACK"}] {move.uci()}' + '<br>'

        if game.board.is_game_over() or game.board.is_fifty_moves(): return Game.get_game_result(game.board)
        
        return move_history
    
    return ""

if __name__ == "__main__":
    
    # parse the command line arguments
    # (if no argument -> launch GUI as webpage on localhost:5000)
    # (else: play/watch a game on the command line)

    game_mode_args = {"--player", "-p", "--self", "-s", "--sunfish", "-sf", "--model", "-m"}

    total_game_mode_args = sum((1 for arg in argv if arg in game_mode_args))

    if total_game_mode_args > 1: 
        print(f"Please provide at most 1 game mode option (found {total_game_mode_args})!")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--player", "-p", action="store_true", help="Start a new game between PyChessBot vs. a player (you)")
        parser.add_argument("--self", "-s", action="store_true", help="Let PyChessBot play a game against itself")
        parser.add_argument("--sunfish", "-sf", action="store_true", help="Let PyChessBot play a game against the Sunfish engine")
        parser.add_argument("--model", "-m", nargs=2, metavar=("model1", "model2"), type=str, help="Let two models from pychessbot/model/ play against each other")
        parser.add_argument("--depth", "-d", metavar="N", type=int, help="search depth for best move prediction")

        args = parser.parse_args()

        if args.depth: Game.depth = args.depth

        if total_game_mode_args == 0: app.run(host="0.0.0.0", port=5000)

        else:
            game = Game("chess_model_v2")

            if args.player: game.play_vs_player()
            elif args.self: game.play_vs_self()
            elif args.model: game.play_vs_model(args.model[1], args.model[0])
            elif args.sunfish: game.play_vs_sunfish()

    Game.update_svg_board(None)
    Game.update_move_history(None, None, None)

        

