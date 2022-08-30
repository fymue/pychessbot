FROM tensorflow/tensorflow
RUN pip install chess
RUN pip install Flask
COPY src/ /pychessbot/src
COPY data/ /pychessbot/data
COPY model/ /pychessbot/model
WORKDIR /pychessbot/src
