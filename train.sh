export BATCH_SIZE=16
#export BATCH_SIZE=1
#export MODEL=MinkowskiTransformerNet
#export MODEL=Res16UNet34C
#export MODEL=MixedTransformer
export MODEL=PointTransformer
#export MODEL=MinkowskiVoxelTransformer

export DATASET=ScannetDataset
#export DATASET=ScannetSparseVoxelizationDataset

export MAX_ITER=12000
export POINTS=4096
export LR=1e-1
#export LR=2e-1
export LOG=$1

#export RESUME=True
./run.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"

