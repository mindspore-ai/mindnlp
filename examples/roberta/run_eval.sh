#!/bin/bash
dataset=(cola mrpc qnli qqp sst2)

for ((i=0; i<${#dataset[*]}; i++))
do
  echo ${dataset[$i]} evaluating
  python roberta_eval.py -c config/${dataset[$i]}_config.yaml
done
