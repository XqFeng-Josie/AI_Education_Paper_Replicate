#!/bin/bash

task=${1:-"en_es"} # en_es, es_en, fr_en
split=${2:-"test"}

wc -l dataset/${task}.slam.20190204.train
echo "--------------------------------------"
wc -l dataset/${task}.slam.20190204.dev
echo "--------------------------------------"
wc -l dataset/${task}.slam.20190204.test
echo "--------------------------------------"

python starter_code/baseline.py \
    --train dataset/${task}.slam.20190204.train \
    --test dataset/${task}.slam.20190204.${split} \
    --pred dataset/${task}.slam.20190204.${split}.pred

python starter_code/eval.py \
    --pred dataset/${task}.slam.20190204.${split}.pred \
    --key dataset/${task}.slam.20190204.${split}.key