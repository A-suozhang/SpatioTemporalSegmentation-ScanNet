export BATCH_SIZE=8
export OMP_NUM_THREADS=1
#export BATCH_SIZE=12
#export BATCH_SIZE=1
#export MODEL=MinkowskiTransformerNet
#export MODEL=Res16UNet34C
#export MODEL=MixedTransformer
#export MODEL=PointTransformer
export MODEL=MinkowskiVoxelTransformer

#export DATASET=ScannetDataset
export DATASET=ScannetSparseVoxelizationDataset
export OPTIMIZER=SGD
export MAX_ITER=24000
#export MAX_ITER=16000
#export POINTS=4096
#export LR=7.5e-
export LR=1e-1
export LOG=$1

#export RESUME=True
./run.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"

