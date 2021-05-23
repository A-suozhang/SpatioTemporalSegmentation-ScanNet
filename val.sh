export BATCH_SIZE=8
export MODEL=MinkowskiVoxelTransformer
#export MODEL=PointTransformer
export DATASET=ScannetSparseVoxelizationDataset
#export MODEL=Res16UNet34C
export MAX_ITER=5000
export LR=1e-1
export LOG=$1
#export DATASET=ScannetVoxelizationDataset
./run_val.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"
