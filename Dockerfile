FROM tensorflow/tensorflow
RUN pip install chess
COPY src/ /pychessbot/src
COPY data/ /pychessbot/data
COPY model/ /pychessbot/model
WORKDIR /pychessbot
