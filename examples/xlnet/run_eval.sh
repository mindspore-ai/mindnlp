#!/bin/bash
dataset=(cola qnli qqp sst2)

for ((i=0; i<${#dataset[*]}; i++))
do
  python xlnet_eval.py -c config/${dataset[$i]}_config.yaml
done