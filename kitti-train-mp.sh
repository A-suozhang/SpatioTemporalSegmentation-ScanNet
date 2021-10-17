export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

export BATCH_SIZE=2
export ITER_SIZE=1
#export BATCH_SIZE=8

export MODEL=Res16UNetTestA
#export MODEL=Res16UNet

export OPTIMIZER=SGD
#export OPTIMIZER=Adam

#export DATASET=ScannetDataset
#export DATASET=ScannetSparseVoxelizationDataset
export DATASET=SemanticKITTI

#export MAX_ITER=12000
export MAX_ITER=30000
#export MAX_ITER=10000
#export MAX_ITER=15000
export MAX_POINTS=195000  # bs=4, h=4 model
#export POINTS=4096
#export LR=7.5e-2

export LR=2e-3
#export LR=1e-1
export MAX_ITER=15000
#export MAX_POINTS=180000  # bs=4, cont-attn
export MAX_POINTS=260000  # bs=4, h=4 model
#export POINTS=4096
#export LR=7.5e-2

export MP=True

#export VOXEL_SIZE=0.1
export VOXEL_SIZE=0.05
export WEIGHT_DECAY=1.e-4

export LOG=$1

#export USE_AUX=True
#export RESUME=True

#export DISTILL=True

./run.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"

