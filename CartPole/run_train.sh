#!/usr/bin/env bash
logdir="logdir"
python="python3"

if [ -d "$logdir" ]; then
    rm -rf "$logdir"
fi

mkdir "$logdir"

$python train.py
