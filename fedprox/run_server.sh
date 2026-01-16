#!/bin/bash

# FedProx 서버 실행 스크립트

echo "================================"
echo "FedProx Server Starting"
echo "================================"

python server.py \
    --num_rounds 20 \
    --num_clients 3 \
    --alpha 0.5 \
    --mu 0.01 \
    --port 8080 \
    --batch_size 128

echo "================================"
echo "Server Finished"
echo "================================"
