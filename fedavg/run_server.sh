#!/bin/bash

# FedAvg 서버 실행 스크립트

echo "================================"
echo "FedAvg Server Starting"
echo "================================"

python server.py \
    --num_rounds 20 \
    --num_clients 3 \
    --alpha 0.5 \
    --port 8080 \
    --batch_size 128

echo "================================"
echo "Server Finished"
echo "================================"
