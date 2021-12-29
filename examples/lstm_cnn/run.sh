#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../../

python examples/lstm_cnn/lstm_cnn_train.py -c examples/lstm_cnn/config/conll2003_config.yaml
python examples/lstm_cnn/lstm_cnn_eval.py -c examples/lstm_cnn/config/conll2003_config.yaml
