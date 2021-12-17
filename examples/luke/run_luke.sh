#!/bin/bash

python train.py -c config/luke_squad.yaml
python eval.py -c config/luke_squad.yaml