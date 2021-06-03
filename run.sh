#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1
export EXPERIMENT=$2
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export DATASET=${DATASET:-ScannetSparseVoxelizationDataset}
export MODEL=${MODEL:-Res16UNet34C}
export OPTIMIZER=${OPTIMIZER:-SGD}
export LR=${LR:-1e-2}
export BATCH_SIZE=${BATCH_SIZE:-12}
export SCHEDULER=${SCHEDULER:-SquaredLR}
export MAX_ITER=${MAX_ITER:-60000}

export RESUME=${RESUME:-none}
export POINTS=${POINTS:-8192}

export USE_AUX=${USE_AUX:-false}
export DISTILL=${DISTILL:-false}

export OUTPATH=./outputs/$DATASET/$MODEL/$LOG/
#export VERSION=$(git rev-parse HEAD)

# Save the experiment detail and dir to the common log file
mkdir -p $OUTPATH

# put the arguments on the first line for easy resume
echo "
    --log_dir $OUTPATH \
    --dataset $DATASET \
    --model $MODEL \
    --train_limit_numpoints 1200000 \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --batch_size $BATCH_SIZE \
    --scheduler $SCHEDULER \
    --max_iter $MAX_ITER \
    $3" 

echo Logging output to "$LOG"
#echo $(pwd) >> $LOG
#echo "Version: " $VERSION >> $LOG
#echo "Git diff" >> $LOG
#echo "" >> $LOG
#git diff | tee -a $LOG
#echo "" >> $LOG
#nvidia-smi | tee -a $LOG

rm ./models
ln -s ./models_ ./models

time python -W ignore main.py \
	--log_dir $OUTPATH \
	--dataset $DATASET \
	--model $MODEL \
	--train_limit_numpoints 1200000 \
	--lr $LR \
	--optimizer $OPTIMIZER \
	--batch_size $BATCH_SIZE \
	--scheduler $SCHEDULER \
	--max_iter $MAX_ITER \
	--num_points $POINTS \
	--resume $RESUME \
	--use_aux $USE_AUX \
	--distill $DISTILL \
	$3 

#time python -W ignore main.py \
    #--log_dir $OUTPATH \
    #--test_config ${OUTPATH}config.json \
    #--weights ${OUTPATH}weights.pth \
    #$3 
	##2>&1 | tee -a "$LOG"
