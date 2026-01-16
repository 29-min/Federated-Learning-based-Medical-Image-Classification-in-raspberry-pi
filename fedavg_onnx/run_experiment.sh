#!/bin/bash

# FedAvg + ONNX 비교 실험 실행 스크립트

echo "================================"
echo "FedAvg + ONNX Comparison Experiment"
echo "================================"

# 1단계: 데이터 전처리 (필요한 경우)
if [ ! -f "client_0_data.pt" ]; then
    echo "데이터 전처리 필요: utils/preprocess_iid_data.py를 먼저 실행하세요"
    echo "cd ../utils && python preprocess_iid_data.py --num_clients 3"
    exit 1
fi

echo "================================"
echo "Server Starting"
echo "================================"

python server.py \
    --num_rounds 20 \
    --min_clients 3 \
    --experiment_name onnx_comparison \
    --batch_size 16

echo "================================"
echo "Experiment Finished"
echo "================================"
