#!/bin/bash

task=${1:-"en_es"} # en_es, es_en, fr_en
split=${2:-"test"}
data_dir=data_${task}

python starter_code/baseline.py \
    --train ${data_dir}/${task}.slam.20190204.train \
    --test ${data_dir}/${task}.slam.20190204.${split} \
    --pred ${data_dir}/baseline_${task}_${split}.pred

python starter_code/eval.py \
    --pred ${data_dir}/baseline_${task}_${split}.pred \
    --key ${data_dir}/${task}.slam.20190204.${split}.key