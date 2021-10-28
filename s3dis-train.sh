export OMP_NUM_THREADS=1

#export BATCH_SIZE=16
#export BATCH_SIZE=6
export BATCH_SIZE=32
#export MODEL=MinkowskiTransformerNet

export MODEL=Res16UNet

#export OPTIMIZER=SGD
export OPTIMIZER=Adam

#export DATASET=ScannetDataset
#export DATASET=ScannetSparseVoxelizationDataset
export DATASET=S3DIS

#export MAX_ITER=12000
export MAX_ITER=24000
export MAX_POINTS=120000
#export POINTS=4096
#export LR=7.5e-2
export LR=1e-3
#export LR=1e-1
#export MP=True

export VOXEL_SIZE=0.01

export WEIGHT_DECAY=1.e-4

export LOG=$1

#export USE_AUX=True
#export RESUME=True
#export DISTILL=True

./run.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"
