#!/bin/bash
dataset=(ag_news dbpedia yelp_p)

for ((i=0; i<${#dataset[*]}; i++))
do
  python dpcnn_train.py -c config/${dataset[$i]}_config.yaml
  python dpcnn_eval.py -c config/${dataset[$i]}_config.yaml
done