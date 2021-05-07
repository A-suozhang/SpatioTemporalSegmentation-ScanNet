import MinkowskiEngine as ME
import torch
import torch.nn as nn

'''
# interp0 = ME.MinkowskiInterpolation()
downsample = ME.MinkowskiSumPooling(kernel_size=1, stride=2, dimension=3)
upsample = ME.MinkowskiPoolingTranspose(kernel_size=1, stride=2, dimension=3)

# make data
# 1st col is batch-id
# coords = torch.tensor([[0, 1., 2., 3.],
                       # [0, 1., 3., 5.],
                       # [0, 3., 4., 8.]
                       # ])
# feats = torch.rand([3,3])

N_voxel=100
feat_dim=16

coords = torch.rand([N_voxel, 3])
coords = torch.cat([torch.zeros(N_voxel,1), coords], dim=1)
feats = torch.rand([N_voxel, feat_dim])

coords, feats = ME.utils.sparse_quantize(coords, feats, quantization_size=0.05)
x = ME.SparseTensor(coordinates=coords, features=feats, device='cuda:0')


y0 = downsample(x)
y1 = upsample(y0)

'''

feats = torch.tensor([
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
    ])

N_voxel = 9
coords = torch.rand([N_voxel, 3])*100
coords = torch.cat([torch.zeros(N_voxel,1), coords], dim=1)

# coords = torch.tensor([[0, 1., 2., 3.],
                       # [0, 1., 3., 5.],
                       # [0, 3., 4., 8.]
                       # ])

conv1 = ME.MinkowskiConvolution(1,1,kernel_size=1,dimension=3)
conv2 = nn.Conv2d(1,1,kernel_size=1,bias=False)

for n, m in conv1.named_parameters():
    print(n,m)

conv2.weight = nn.Parameter(m.reshape(1,1,1,1))

x1 = ME.SparseTensor(coordinates=coords, features=feats.reshape([9,1]))
y1 = conv1(x1)

y2 = conv2(x1.F.reshape(1,1,3,3))

# they are the same








