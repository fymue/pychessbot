# pychessbot

A simple neural network that knows how to play chess written in Python.

## What data was used for training?

The data I used to train the model consists of a bunch of Grandmaster games taken from [this website](https://www.pgnmentor.com/files.html).
All of the games I used to train the model can be found under [data](https://github.com/fymue/pychessbot/tree/main/data).

## How was the model trained?

In oder to enable learning how to play chess, I went through all moves of Checkmate games, converted all resulting board states into 8x8x6 tensors and labeled the board states resulting from a move of the winner of the game as a "good" board state (meaning the last move was a good move). Since I only used Grandmaster games as training data though, the loser's moves weren't necessarily "bad" moves (up until a certain point). So instead of labeling the loser's moves as "bad" moves, I let the program generate a random (legal) move as a replacement for a "bad" move. That way the model learns to differentiate between good and bad moves/board states.

## How does it predict the next (best) move to play?

To predict the next move the bot should play, the program generates all possible (legal) moves whenever the model is at turn and and predicts an output value for all board states resulting from those moves. It then either picks the move that yielded the highest output value or performs additional alpha beta pruning to a user-specified depth to further evaluate the search tree and to find a (potentially) better move.

## Installation (Linux)

First, you obviously need to download the repository.
```
git clone https://github.com/fymue/pychessbot.git && cd pychessbot
``` 

### Docker/Podman

The most convenient way to install/run this project is to run it inside a container. This way you don't have to (globally) install any dependencies. In order to do this you need to have either docker or podman installed on your system. If that is the case, after cloning the repository you can simply build a docker container image, which will contain all required dependencies to run the program, by executing the command
```
(docker|podman) build -t imagename .
```
After that you can use this image to build and (interactively) execute a container, from which the program can be run.
```
(docker|podman) run -it imagename bash
```
Inside this container you can now run the program without having to globally install any dependencies. 

### From source

If you prefer to install the program from source instead, you can use the <code>requirements.txt</code> file to install all required packages.
```
pip install -r requirements.txt
```

## Usage

The program can either be run from the command line or as a web application in your browser (on localhost:5000) by executing the command
```
python3 src/play.py [-h] [--player] [--self] [--sunfish] [--model model1 model2]
```
