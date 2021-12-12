#!/bin/bash
dataset=(cola qnli qqp sst2)

for ((i=0; i<${#dataset[*]}; i++))
do
  python xlnet_train.py -c config/${dataset[$i]}_config.yaml
done