export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
#export BATCH_SIZE=16
#export BATCH_SIZE=6
#export BATCH_SIZE=8
export BATCH_SIZE=4
#export MODEL=MinkowskiTransformerNet
#export MODEL=Res16UNet34C
#export MODEL=Res16UNet18A
#export MODEL=MixedTransformer
#export MODEL=PointTransformer
#export MODEL=MinkowskiVoxelTransformer

export MODEL=Res16UNetTestA
#export MODEL=Res16UNet

export OPTIMIZER=SGD
#export DATASET=ScannetDataset
#export DATASET=ScannetSparseVoxelizationDataset
export DATASET=SemanticKITTI

#export MAX_ITER=12000
export MAX_ITER=15000
export MAX_POINTS=2400000
#export POINTS=4096
#export LR=7.5e-2
export LR=5e-2

#export MP=True

export VOXEL_SIZE=0.1
export WEIGHT_DECAY=1.e-4

export LOG=$1

#export USE_AUX=True
#export RESUME=True

#export DISTILL=True

./run.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"

