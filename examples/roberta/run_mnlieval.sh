#!/bin/bash

do
  echo mnli evaluating
  python roberta_mnlieval.py -c config/mnli_config.yaml
done

