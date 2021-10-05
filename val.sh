export BATCH_SIZE=2
export OMP_NUM_THREADS=1

#export MODEL=Res16UNet34C
#export MODEL=Res16UNet18A
#export MODEL=Res16UNetTestA
export MODEL=Res16UNet

#export MODEL=PointTransformer
#export IS_EXPORT=True
#export DATASET=ScannetDataset
#export DATASET=ScannetSparseVoxelizationDataset
#export DATASET=ScannetSparseVoxelizationDataset
export DATASET=SemanticKITTI

export MAX_ITER=5000
export LR=1e-1
export LOG=$1
#export DATASET=ScannetVoxelizationDataset
./run_val.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"
