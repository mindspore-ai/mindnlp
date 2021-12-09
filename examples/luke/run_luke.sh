#!/bin/bash
export PYTHONPATH=../../

python train.py -c config/luke_squad.yaml
python eval.py -c config/luke_squad.yaml