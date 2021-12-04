#!/bin/bash
dataset=(cola mnli mrpc qnli qqp sst2)

for ((i=0; i<${#dataset[*]}; i++))
do
  python roberta_train.py -c config/${dataset[$i]}_config.yaml
  python roberta_eval.py -c config/${dataset[$i]}_config.yaml
done