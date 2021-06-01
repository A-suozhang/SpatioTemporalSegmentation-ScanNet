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
        # [4., 5., 6.],
        # [7., 8., 9.],
    ])
feats = nn.Parameter(feats)

feats2 = torch.tensor([
        [0., 2., 3.],
        # [4., 5., 6.],
        # [7., 8., 9.],
    ])
feats2 = nn.Parameter(feats2)


N_voxel = 3
# coords = torch.rand([N_voxel, 3])*10
# coords = torch.rand([N_voxel])*10
# coords = torch.cat([torch.zeros(N_voxel,1), coords], dim=1)

coords = torch.tensor([[0, 2., 4., 2.],
                       [0, 1., 3., 1.],
                       [0, 0., 3., 1.]
                       ])

conv1 = ME.MinkowskiConvolution(1,1,kernel_size=1,dimension=3)
conv2 = nn.Conv2d(1,1,kernel_size=1,bias=False)


conv_s2 = ME.MinkowskiConvolution(1,1,kernel_size=2,stride=2,dimension=3)
pool_s2 = ME.MinkowskiMaxPooling(2,stride=2,dimension=3)



for n, m in conv1.named_parameters():
    print(n,m)

conv2.weight = nn.Parameter(m.reshape(1,1,1,1))

x1 = ME.SparseTensor(coordinates=coords, features=feats.reshape([-1,1]))
x2 = ME.SparseTensor(coordinates=coords, features=feats2.reshape([-1,1]))

x1_s2 = conv_s2(x1)
x1_p2 = pool_s2(x1)

'''
a little demo of getting the neighbor of each point manually after stride=2 conv
'''

batch_id = x1_p2.C[:,0] # [N,1]
batch_id = batch_id.unsqueeze(-1).repeat(1,8).reshape(-1,1) # [N*8, 1]

pooled_C = x1_p2.C[:,1:]
kernel_C = pooled_C.unsqueeze(1).repeat(1,8,1) # [N, ks, 3]

diffs = torch.tensor([
        [0,0,0],
        [0,0,1],
        [0,1,0],
        [0,1,1],
        [1,0,0],
        [1,0,1],
        [1,1,0],
        [1,1,1],
    ])

kernel_C = kernel_C + diffs

query_C = torch.cat([batch_id, kernel_C.reshape(-1,3)], dim=1)
query_F = x1.features_at_coordinates(query_C.float()).reshape(-1,8)  # [N, 8]

# find the most freq
out, _ = torch.mode(query_F)

import ipdb; ipdb.set_trace()

subsampled_x1 = x1.features_at_coordinates(x1_s2.C.float())


# part_x1_feat = x1.features_at_coordinates(x1.C[:3,:].float())
# part_x1_coord = x1.C[:3,:]

# part_x1 = ME.SparseTensor(
        # coordinates = part_x1_coord,
        # features = part_x1_feat,
        # coordinate_manager = x1.coordinate_manager,
        # )
# out = part_x1 + x1  # out and x1 may not have the same order
# x2_f = x2.F
# x1_f = x1_f**2 + x2_f**2
# y = ME.SparseTensor(features = x1_f, coordinate_map_key=x1.coordinate_map_key, coordinate_manager=x1.coordinate_manager)

pool = ME.MinkowskiGlobalSumPooling()

loss = pool(out).F
loss.backward()

# # if there isnot nearby voxel within kernel size, then no spatial reduction will occur
# # pool = ME.MinkowskiSumPooling(kernel_size=6,stride=2,dimension=3)

# loss = pool(y).F
# loss.backward()

print(feats, feats.grad)










