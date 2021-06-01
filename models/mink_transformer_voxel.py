# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import random
import numpy as np
import glob

try:
    import h5py
except:
    print("Install h5py with `pip install h5py`")
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me

from models.model import Model, NetworkType
from models.modules.common import ConvType, NormType, get_norm, conv, sum_pool
from models.modules.resnet_block import BasicBlock, Bottleneck

import sys
# sys.path.append('/home/zhaotianchen/project/point-transformer/pt-cls/model')
from models.pct_voxel_utils import TDLayer, TULayer, PTBlock, StackedPTBlock



def subsample_aux(x0, x1, aux, kernel_size=2):

    aux = ME.SparseTensor(coordinates=aux.C, features=(aux.F+1))

    n_ks = kernel_size**3

    coords = x1.C
    batch_id = coords[:,0]
    batch_id = batch_id.unsqueeze(-1).repeat(1,n_ks).reshape(-1,1)

    pooled_C = coords[:,1:]
    kernel_C = pooled_C.unsqueeze(1).repeat(1,n_ks,1)

    diffs = torch.tensor([
            [0,0,0],
            [0,0,1],
            [0,1,0],
            [0,1,1],
            [1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1],
        ], device='cuda')

    kernel_C = kernel_C + diffs

    query_C = torch.cat([batch_id, kernel_C.reshape(-1,3)], dim=1).float()
    query_F = aux.features_at_coordinates(query_C).reshape(-1,8)  # [N, 8]

    # find the most freq: find normal will always return 0
    # out, _ = torch.mode(query_F)

    freqs = torch.stack([(query_F == i).sum(dim=1) for i in range(1,21)])
    out = freqs.max(dim=0)[1] # shape [N]

    out = out - 1 # [-1~20]

    # debug dict for plot
    # d = {}
    # d['origin_pc'] = x0.C
    # d['origin_pred'] = aux.F
    # d['new_pc'] = x1.C
    # d['new_pred'] = out
    # torch.save(d, './aux.pth')

    out = ME.SparseTensor(coordinates=x1.C, features=out.float().reshape([-1,1]).cuda())

    return out

class MinkowskiVoxelTransformer(ME.MinkowskiNetwork):

    def __init__(self, config, in_channel, out_channel, final_dim=96, dimension=3):

        ME.MinkowskiNetwork.__init__(self, dimension)
        # The normal channel for Modelnet is 3, for scannet is 6, for scanobjnn is 0
        normal_channel = 3

        self.CONV_TYPE = ConvType.SPATIAL_HYPERCUBE

        self.dims = np.array([32, 64, 128, 256])
        self.neighbor_ks = np.array([12, 12, 12, 12])
        # self.neighbor_ks = np.array([16, 16, 16, 16])
        # self.neighbor_ks = np.array([8, 8, 8, 8])

        self.final_dim = final_dim

        stem_dim =  self.dims[0]

        if config.xyz_input:
            in_channel = normal_channel + in_channel
        else:
            in_channel = in_channel

        # the 1st ds & last upsample use stridedConv/TransitionUp/Down
        self.POINT_TR_LIKE = False

        # pixel size 1
        self.stem1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channel, stem_dim, kernel_size=config.ks, dimension=3),
            ME.MinkowskiBatchNorm(stem_dim),
            ME.MinkowskiReLU(),
        )

        if self.POINT_TR_LIKE:
            self.stem2 = TDLayer(input_dim=self.dims[0], out_dim=self.dims[0])
        else:
            # does the spatial downsampling
            # pixel size 2
            self.stem2 = nn.Sequential(
                    ME.MinkowskiConvolution(stem_dim, stem_dim, kernel_size=config.ks, dimension=3, stride=2),
                    ME.MinkowskiBatchNorm(stem_dim),
                    ME.MinkowskiReLU(),
                )

        base_r = 5

        self.PTBlock1 = PTBlock(in_dim=self.dims[0], hidden_dim = self.dims[0], n_sample=self.neighbor_ks[0], skip_knn=False, r=base_r, kernel_size=config.ks)
        self.PTBlock2 = PTBlock(in_dim=self.dims[1], hidden_dim = self.dims[1], n_sample=self.neighbor_ks[1], skip_knn=False, r=2*base_r, kernel_size=config.ks)
        self.PTBlock3 = PTBlock(in_dim=self.dims[2],hidden_dim = self.dims[2], n_sample=self.neighbor_ks[2], skip_knn=False, r=2*base_r, kernel_size=config.ks)
        self.PTBlock4 = PTBlock(in_dim=self.dims[3], hidden_dim = self.dims[3], n_sample=self.neighbor_ks[3], skip_knn=False, r=4*base_r, kernel_size=config.ks)
        self.PTBlock5 = PTBlock(in_dim=self.dims[2], hidden_dim = self.dims[2], n_sample=self.neighbor_ks[3], skip_knn=False, r=2*base_r, kernel_size=config.ks) # out: 256
        self.PTBlock6 = PTBlock(in_dim=self.dims[1], hidden_dim=self.dims[1], n_sample=self.neighbor_ks[2], skip_knn=False, r=2*base_r, kernel_size=config.ks) # out: 128
        self.PTBlock7 = PTBlock(in_dim=self.dims[0], hidden_dim=self.dims[0], n_sample=self.neighbor_ks[1], skip_knn=False, r=base_r, kernel_size=config.ks) # out: 64

        # self.PTBlock1 = StackedPTBlock(in_dim=self.dims[0], hidden_dim = self.dims[0], n_sample=self.neighbor_ks[0], skip_knn=False, r=base_r, kernel_size=config.ks)
        # self.PTBlock2 = StackedPTBlock(in_dim=self.dims[1], hidden_dim = self.dims[1], n_sample=self.neighbor_ks[1], skip_knn=False, r=2*base_r, kernel_size=config.ks)
        # self.PTBlock3 = StackedPTBlock(in_dim=self.dims[2],hidden_dim = self.dims[2], n_sample=self.neighbor_ks[2], skip_knn=False, r=2*base_r, kernel_size=config.ks)
        # self.PTBlock4 = StackedPTBlock(in_dim=self.dims[3], hidden_dim = self.dims[3], n_sample=self.neighbor_ks[3], skip_knn=False, r=4*base_r, kernel_size=config.ks)
        # self.PTBlock5 = StackedPTBlock(in_dim=self.dims[2], hidden_dim = self.dims[2], n_sample=self.neighbor_ks[3], skip_knn=False, r=2*base_r, kernel_size=config.ks) # out: 256
        # self.PTBlock6 = StackedPTBlock(in_dim=self.dims[1], hidden_dim=self.dims[1], n_sample=self.neighbor_ks[2], skip_knn=False, r=2*base_r, kernel_size=config.ks) # out: 128
        # self.PTBlock7 = StackedPTBlock(in_dim=self.dims[0], hidden_dim=self.dims[0], n_sample=self.neighbor_ks[1], skip_knn=False, r=base_r, kernel_size=config.ks) # out: 64

        # self.PTBlock1 = self._make_layer(block=BasicBlock, inplanes=self.dims[0], planes=self.dims[0], num_blocks=2)
        # self.PTBlock2 = self._make_layer(block=BasicBlock, inplanes=self.dims[1], planes=self.dims[1], num_blocks=2)
        # self.PTBlock3 = self._make_layer(block=BasicBlock, inplanes=self.dims[2], planes=self.dims[2], num_blocks=2)
        # self.PTBlock4 = self._make_layer(block=BasicBlock, inplanes=self.dims[3], planes=self.dims[3], num_blocks=2)
        # self.PTBlock5 = self._make_layer(block=BasicBlock, inplanes=self.dims[2], planes=self.dims[2], num_blocks=2)
        # self.PTBlock6 = self._make_layer(block=BasicBlock, inplanes=self.dims[1], planes=self.dims[1], num_blocks=2)
        # self.PTBlock7 = self._make_layer(block=BasicBlock, inplanes=self.dims[0], planes=self.dims[0], num_blocks=2)

        # pixel size 2
        self.TDLayer1 = TDLayer(input_dim=self.dims[0], out_dim=self.dims[1]) # strided conv

        # pixel size 4
        self.TDLayer2 = TDLayer(input_dim=self.dims[1], out_dim=self.dims[2])

        # pixel size 8
        self.TDLayer3 = TDLayer(input_dim=self.dims[2], out_dim=self.dims[3])

        # pixel size 16: PTBlock4

        # pixel size 8
        self.TULayer5 = TULayer(input_a_dim=self.dims[3], input_b_dim = self.dims[2], out_dim=self.dims[2]) # out: 256//2 + 128 = 256

        # pixel size 4
        self.TULayer6 = TULayer(input_a_dim=self.dims[2], input_b_dim = self.dims[1], out_dim=self.dims[1]) # out: 256//2 + 64 = 192

        # pixel size 2
        self.TULayer7 = TULayer(input_a_dim=self.dims[1], input_b_dim = self.dims[0], out_dim=self.dims[0]) # 128 // 2 + 32 = 96

        # pixel size 1
        # self.PTBlock8 = PTBlock(in_dim=self.dims[0], hidden_dim=self.dims[0], n_sample=self.neighbor_ks[1])  # 32

        if self.POINT_TR_LIKE:
            self.TULayer8 = TULayer(input_a_dim=self.dims[0], input_b_dim = self.dims[0], out_dim=self.dims[0]) # 64 // 2 + 32
            self.fc = ME.MinkowskiLinear(self.dims[0], out_channel)
        else:
            self.final_conv = nn.Sequential(
                ME.MinkowskiConvolutionTranspose(self.dims[0], self.final_dim, kernel_size=2, stride=2, dimension=3),
                ME.MinkowskiBatchNorm(self.final_dim),
                ME.MinkowskiReLU(),
                # ME.MinkowskiDropout(0.4), # DEBUG: no use
            )
            self.fc = ME.MinkowskiLinear(self.final_dim+self.dims[0], out_channel)

    def forward(self, in_field: ME.TensorField, aux=None):

        x = in_field

        if aux is not None:
            aux = ME.SparseTensor(coordinates=x.C, features=aux.float().reshape([-1,1]).cuda())

            x0 = self.stem1(x)
            x = self.stem2(x0)

            aux1 = subsample_aux(x0,x,aux)
            x1 = self.PTBlock1(x, aux=aux1)

            x = self.TDLayer1(x1)
            aux2 = subsample_aux(x1,x,aux1)
            x2 = self.PTBlock2(x, aux=aux2)

            x = self.TDLayer2(x2)
            aux3 = subsample_aux(x2,x,aux2)
            x3 = self.PTBlock3(x, aux=aux3)

            x = self.TDLayer3(x3)
            aux4 = subsample_aux(x3,x,aux3)
            x = self.PTBlock4(x, aux=aux4)

            x = self.TULayer5(x, x3)
            x = self.PTBlock5(x, aux=aux3)

            x = self.TULayer6(x, x2)
            x = self.PTBlock6(x, aux=aux2)

            x = self.TULayer7(x, x1)
            x = self.PTBlock7(x, aux=aux1)

        else:
            x0 = self.stem1(x)
            x = self.stem2(x0)

            x1 = self.PTBlock1(x)

            x = self.TDLayer1(x1)
            x2 = self.PTBlock2(x)

            x = self.TDLayer2(x2)
            x3 = self.PTBlock3(x)

            x = self.TDLayer3(x3)
            x = self.PTBlock4(x)

            x = self.TULayer5(x, x3)
            x = self.PTBlock5(x)

            x = self.TULayer6(x, x2)
            x = self.PTBlock6(x)

            x = self.TULayer7(x, x1)
            x = self.PTBlock7(x)

        if self.POINT_TR_LIKE:
            x = self.TULayer8(x, x0)
            x = self.fc(x)
        else:
            x = self.final_conv(x)
            x = self.fc(me.cat(x0,x))

        return x

    def _make_layer(self,
                  block,
                  inplanes,
                  planes,
                  num_blocks,
                  stride=1,
                  dilation=1,
                  norm_type=NormType.BATCH_NORM,
                  nonlinearity_type='ReLU',
                  bn_momentum=0.1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
          downsample = nn.Sequential(
              conv(
                  inplanes,
                  planes * block.expansion,
                  kernel_size=1,
                  stride=stride,
                  bias=False,
                  D=self.D),
              get_norm(norm_type, planes * block.expansion, D=self.D, bn_momentum=bn_momentum),
          )
        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_type=self.CONV_TYPE,
                nonlinearity_type=nonlinearity_type,
                D=self.D))

        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
          layers.append(
              block(
                  inplanes,
                  planes,
                  stride=1,
                  dilation=dilation,
                  conv_type=self.CONV_TYPE,
                  nonlinearity_type=nonlinearity_type,
                  D=self.D))

        return nn.Sequential(*layers)





