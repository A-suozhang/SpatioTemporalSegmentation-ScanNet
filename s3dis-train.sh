export OMP_NUM_THREADS=1

#export BATCH_SIZE=16
#export BATCH_SIZE=6
export BATCH_SIZE=2
export ITER_SIZE=2
#export MODEL=MinkowskiTransformerNet
#export MODEL=Res16UNet34C
#export MODEL=Res16UNet18A
export MODEL=Res16UNetTestA
#export MODEL=MixedTransformer
#export MODEL=PointTransformer
#export MODEL=MinkowskiVoxelTransformer

#export OPTIMIZER=SGD
export OPTIMIZER=Adam

#export DATASET=ScannetDataset
export DATASET=S3DIS
#export DATASET=ScannetSparseVoxelizationDataset

#export MAX_ITER=12000
export MAX_ITER=3000
#export MAX_ITER=48000
export MAX_POINTS=2500000   # voxel-size 0.02
#export MAX_POINTS=3000000   # voxel-size 0.02
#export POINTS=4096
#export LR=7.5e-2
#export LR=1e-3
#export LR=5e-2
export LR=1e-3
export MP=True

#export IS_DEBUG=True

export VOXEL_SIZE=0.02
export WEIGHT_DECAY=1.e-4

export LOG=$1

#export USE_AUX=True
#export RESUME=True
#export DISTILL=True

#./run_val.sh $2 \
./run.sh $2 \
		-default \
		"--scannet_path /data/eva_share_users/zhaotianchen/scannet_processed/train"

