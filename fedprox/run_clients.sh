#!/bin/bash

# FedProx 클라이언트 실행 스크립트
# 사용법: ./run_clients.sh [서버주소]

SERVER_ADDRESS=${1:-"192.168.45.100:8080"}
TOTAL_CLIENTS=3
ALPHA=0.5
MU=0.01

echo "================================"
echo "FedProx Client Launcher"
echo "================================"
echo "Server: $SERVER_ADDRESS"
echo "Total Clients: $TOTAL_CLIENTS"
echo "Alpha: $ALPHA"
echo "Mu (proximal term): $MU"
echo "================================"
echo ""
echo "다음 명령어를 각 터미널에서 실행하세요:"
echo ""

for i in $(seq 0 $((TOTAL_CLIENTS-1))); do
    echo "터미널 $((i+1)):"
    echo "python client.py --client_id $i --total_clients $TOTAL_CLIENTS --alpha $ALPHA --mu $MU --server_address $SERVER_ADDRESS"
    echo ""
done

echo "================================"
