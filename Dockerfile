FROM tensorflow/tensorflow
COPY src/ /pychessbot/src
COPY data/ /pychessbot/data
WORKDIR /pychessbot
RUN pip install chess
