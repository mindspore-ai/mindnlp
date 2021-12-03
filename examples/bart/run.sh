#!/bin/bash
dataset=(xsum)

for ((i=0; i<${#dataset[*]}; i++))
do
  python bart_train.py -c config/${dataset[$i]}_config.yaml
  python bart_eval.py -c config/${dataset[$i]}_config.yaml
done