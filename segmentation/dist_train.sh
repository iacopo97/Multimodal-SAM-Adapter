#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29301}

CUDA_VISIBLE_DEVICES=1,2  python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --deterministic ${@:3} 
