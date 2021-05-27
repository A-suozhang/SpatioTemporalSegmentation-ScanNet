export OMP_NUM_THREADS=1

#export BATCH_SIZE=16
#export BATCH_SIZE=7
export BATCH_SIZE=8
#export MODEL=MinkowskiTransformerNet
#export MODEL=Res16UNet34C
#export MODEL=Res16UNet18A
#export MODEL=MixedTransformer
#export MODEL=PointTransformer
export MODEL=MinkowskiVoxelTransformer

#export DATASET=ScannetDataset
export DATASET=ScannetSparseVoxelizationDataset

export MAX_ITER=13000
#export MAX_ITER=16000
#export POINTS=4096
#export LR=7.5e-2
export LR=1e-1
export LOG=$1

export RESUME=True

./run.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"

