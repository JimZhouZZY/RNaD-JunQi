#!/bin/bash

while true; do
    # 获取当前时间，格式为 YYYYMMDD_HHMMSS
    current_time=$(date +"%Y%m%d_%H%M%S")
    
    # 创建以当前时间为名的文件夹
    mkdir -p "data/$current_time"

    # 复制 .png、.txt、.pkl 文件到新文件夹
    cp data/model_auto_saved.pkl data/model.pkl
    cp data/model_auto_saved.pkl "data/$current_time"
    mv data/*.png "data/$current_time"
    cp data/*.txt "data/$current_time"

    # 执行 train.py
    python train.py
done

