FROM docker.io/tensorflow/tensorflow:latest
RUN pip install chess
RUN pip install Flask
COPY src/ /pychessbot/src
COPY data/ /pychessbot/data
COPY model/ /pychessbot/model
WORKDIR /pychessbot/src
CMD ["python3", "play.py"]
