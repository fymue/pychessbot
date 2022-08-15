# pychessbot

A simple neural network that knows how to play chess written in Python.

## What data was used for training?

The data I used to train the model was a bunch of recorded Grandmaster games taken from [this website](https://www.pgnmentor.com/files.html).
All of the games I used to train the model can be found under [data](https://github.com/fymue/pychessbot/tree/main/data).

## How was the model trained?

In oder to enable learning how to play chess, I labeled every move of the winner of a chess game as a good move (meaning a move leading to a win). Since I only used Grandmaster games as training data though, the loser's moves weren't necessarily bad moves (up until a certain point). So instead of labeling the loser's moves as bad moves, I let the program generate a random (legal) move as a replacement for a bad move. That way the model learns to differentiate between good and bad moves and ultimately learns how the play chess.