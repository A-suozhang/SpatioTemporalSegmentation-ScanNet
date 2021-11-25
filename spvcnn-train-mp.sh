export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export BATCH_SIZE=2
export ITER_SIZE=2
export MODEL=Res16UNetTestA
export OPTIMIZER=SGD
#export DATASET=ScannetDataset
export DATASET=ScannetSparseVoxelizationDataset
#export DATASET=SemanticKITTI
export MAX_ITER=30000
export MAX_POINTS=350000 
export LR=0.2
export MP=True
export VOXEL_SIZE=0.02
export WEIGHT_DECAY=1.e-4
export LOG=$1
export ENABLE_POINT_BRANCH=True # SPVCNN feature

./run.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"

