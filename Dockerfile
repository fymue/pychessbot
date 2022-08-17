FROM tensorflow/tensorflow
COPY src/ /pychessbot/src
COPY data/ /pychessbot/data
WORKDIR /pychessbot
CMD ["python3", "src/model.py"]