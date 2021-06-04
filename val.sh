export BATCH_SIZE=8

#export MODEL=Res16UNet34C
#export MODEL=Res16UNet18A
export MODEL=MinkowskiVoxelTransformer

#export MODEL=PointTransformer
#export IS_EXPORT=True
#export DATASET=ScannetDataset
export DATASET=ScannetSparseVoxelizationDataset
export MAX_ITER=5000
export LR=1e-1
export LOG=$1
#export DATASET=ScannetVoxelizationDataset
./run_val.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"
