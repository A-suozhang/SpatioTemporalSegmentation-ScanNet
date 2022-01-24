export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

export BATCH_SIZE=8
export ITER_SIZE=1

export MODEL=Res16UNetTestA

#export OPTIMIZER=SGD
export OPTIMIZER=Adam

export DATASET=SemanticKITTI

export MAX_ITER=15000
export LR=2e-3
export MAX_ITER=15000
#export MAX_POINTS=180000  # bs=4, cont-attn
export MAX_POINTS=200000  # bs=4, h=4 model
#export POINTS=4096

export MP=False
export VOXEL_SIZE=0.05

export WEIGHT_DECAY=1.e-5
#export WEIGHT_DECAY=5.e-6

export LOG=$1

#export USE_AUX=True
export RESUME=False
export SUBMIT=True

#export DISTILL=True
#export ENABLE_POINT_BRANCH=True # SPVCNN feature

./run_val.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"

