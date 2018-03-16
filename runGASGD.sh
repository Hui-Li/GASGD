#!/usr/bin/env bash

rm -rf build
mkdir build
cd build
cmake ../
make

path="../../mf_data/ml10m"
dimension=50
lambda=0.05
g_period=0.01
step=1 # 1: bold driver, 0: constant size
partition=2 # 0: Greedy, 1: Item Partition, 2: User Partition
learning_rate=0.0125
folder=1
node=1
thread=4
max_iter=10

./runGASGD --k $dimension --step $step --partition $partition --node $node \
--thread $thread --g_period $g_period --lr $learning_rate --folder $folder --max_iter $max_iter --path $path