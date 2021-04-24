export BATCH_SIZE=6
export MODEL=MinkowskiTransformerNet
#export MODEL=Res16UNet34C
export MAX_ITER=10000
export LR=1e-1
export LOG=$1
#export DATASET=ScannetVoxelizationDataset
./run.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"

