#!/bin/bash 
set -x 
# THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
# cd $THIS_DIR

ports=(${WORKER_0_PORT//,/ })
port=${ports[0]}

S2S_DIR="YOUR_S2S_DIR"
cd $S2S_DIR/code

#get data
sudo mkdir -p /export/home/.cache/
sudo chmod a+w /export/home/.cache
cd /export/home/.cache/
ln -s ${S2S_DIR}/data lavis
cd -


# You can zip the cache folder and copy it to the worker node. Please be aware this should comply with the usage license of the checkpoint providers.
rm -r ~/.cache
cp ${S2S_DIR}/envs/BLIP.cache.tar ~/
cd ~/
tar xf BLIP.cache.tar
cd -


cd $THIS_DIR

CONDA_ENV_DIR="YOUR_CONDA_ENV_DIR"
NGPUS=$WORKER_GPU
CONFIG=lavis/projects/alpro/train/bdmsvdc_video_retrieval_ft.yaml

$CONDA_ENV_DIR/envs/shot2story/bin/python -m torch.distributed.run --nproc_per_node=$NGPUS --nnode=$WORKER_NUM --node_rank=$WORKER_ID --master_addr=$WORKER_0_HOST --master_port=$port train.py \
--cfg-path $CONFIG $@
