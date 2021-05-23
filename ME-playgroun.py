import MinkowskiEngine as ME
import torch


# interp0 = ME.MinkowskiInterpolation()
downsample = ME.MinkowskiSumPooling(kernel_size=1, stride=2, dimension=3)
upsample = ME.MinkowskiPoolingTranspose(kernel_size=1, stride=2, dimension=3)

stride_conv = ME.MinkowskiConvolution(16, 16, kernel_size=4, stride=2, bias=False, dimension=3)

'''make data'''
# 1st col is batch-id
coords = torch.tensor([[0, 0., 0., 1.],
                       [0, 0., 0., 3.]
                      ])
N_voxel=100
feat_dim=16

# coords = torch.rand([N_voxel, 3])
# coords = torch.cat([torch.zeros(N_voxel,1), coords], dim=1)
feats = torch.rand([2, feat_dim])
feats[0, :] = 0

coords, feats = ME.utils.sparse_quantize(coords, feats, quantization_size=1)
x = ME.SparseTensor(coordinates=coords, features=feats, device='cpu')

stride_y = stride_conv(x)

print("===> x.C")
print(x.C)
print("===> y.C")
print(stride_y.C)
print("===> x.F")
print(x.F)
print("===> y.F")
print(stride_y.F)


y0 = downsample(x)
y1 = upsample(y0)


