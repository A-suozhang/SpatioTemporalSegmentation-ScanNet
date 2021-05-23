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
from models.pct_voxel_utils import TDLayer, TULayer, PTBlock

class MinkowskiVoxelTransformer(ME.MinkowskiNetwork):

    def __init__(self, config, in_channel, out_channel, final_dim=96, dimension=3):

        ME.MinkowskiNetwork.__init__(self, dimension)
        # The normal channel for Modelnet is 3, for scannet is 6, for scanobjnn is 0
        normal_channel = 3

        self.CONV_TYPE = ConvType.SPATIAL_HYPERCUBE

        self.dims = np.array([32, 64, 128, 256])
        self.neighbor_ks = np.array([32, 32, 32, 32, 32]) // 2

        self.final_dim = final_dim

        stem_dim =  self.dims[0]

        if config.xyz_input:
            in_channel = normal_channel + in_channel
        else:
            in_channel = in_channel

        # pixel size 1
        self.stem1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channel, stem_dim, kernel_size=config.ks, dimension=3),
            ME.MinkowskiBatchNorm(stem_dim),
            ME.MinkowskiReLU(),
        )

        # when using split-scnee, no stride here
        # does the spatial downsampling
        # pixel size 2
        self.stem2 = nn.Sequential(
                ME.MinkowskiConvolution(stem_dim, stem_dim, kernel_size=config.ks, dimension=3, stride=2),
                ME.MinkowskiBatchNorm(stem_dim),
                ME.MinkowskiReLU(),
            )


        base_r = 20

        self.PTBlock1 = PTBlock(in_dim=self.dims[0], hidden_dim = self.dims[0], n_sample=self.neighbor_ks[0], skip_knn=False, r=base_r, kernel_size=config.ks)
        self.PTBlock2 = PTBlock(in_dim=self.dims[1], hidden_dim = self.dims[1], n_sample=self.neighbor_ks[1], skip_knn=False, r=2*base_r, kernel_size=config.ks)
        self.PTBlock3 = PTBlock(in_dim=self.dims[2],hidden_dim = self.dims[2], n_sample=self.neighbor_ks[2], skip_knn=False, r=2*base_r, kernel_size=config.ks)

        self.PTBlock4 = PTBlock(in_dim=self.dims[3], hidden_dim = self.dims[3], n_sample=self.neighbor_ks[3], skip_knn=False, r=4*base_r, kernel_size=config.ks)

        self.PTBlock5 = PTBlock(in_dim=self.dims[2], hidden_dim = self.dims[2], n_sample=self.neighbor_ks[3], skip_knn=False, r=2*base_r, kernel_size=config.ks) # out: 256
        self.PTBlock6 = PTBlock(in_dim=self.dims[1], hidden_dim=self.dims[1], n_sample=self.neighbor_ks[2], skip_knn=False, r=2*base_r, kernel_size=config.ks) # out: 128
        self.PTBlock7 = PTBlock(in_dim=self.dims[0], hidden_dim=self.dims[0], n_sample=self.neighbor_ks[1], skip_knn=False, r=base_r, kernel_size=config.ks) # out: 64

        # self.PTBlock1 = PTBlock(in_dim=self.dims[0], hidden_dim = self.dims[0], n_sample=self.neighbor_ks[0], skip_knn=True)
        # self.PTBlock2 = PTBlock(in_dim=self.dims[1], hidden_dim = self.dims[1], n_sample=self.neighbor_ks[1], skip_knn=True)
        # self.PTBlock3 = PTBlock(in_dim=self.dims[2],hidden_dim = self.dims[2], n_sample=self.neighbor_ks[2], skip_knn=True)
        # self.PTBlock4 = PTBlock(in_dim=self.dims[3], hidden_dim = self.dims[3], n_sample=self.neighbor_ks[3], skip_knn=True)
        # self.PTBlock5 = PTBlock(in_dim=self.dims[3], hidden_dim = self.dims[3], n_sample=self.neighbor_ks[3], skip_knn=True) # out: 256
        # self.PTBlock6 = PTBlock(in_dim=self.dims[2], hidden_dim=self.dims[2], n_sample=self.neighbor_ks[2], skip_knn=True) # out: 128
        # self.PTBlock7 = PTBlock(in_dim=self.dims[1], hidden_dim=self.dims[1], n_sample=self.neighbor_ks[1], skip_knn=True) # out: 64

        # self.PTBlock1 = self._make_layer(block=BasicBlock, inplanes=self.dims[0], planes=self.dims[0], num_blocks=2)
        # self.PTBlock2 = self._make_layer(block=BasicBlock, inplanes=self.dims[1], planes=self.dims[1], num_blocks=2)
        # self.PTBlock3 = self._make_layer(block=BasicBlock, inplanes=self.dims[2], planes=self.dims[2], num_blocks=2)
        # self.PTBlock4 = self._make_layer(block=BasicBlock, inplanes=self.dims[3], planes=self.dims[3], num_blocks=2)
        # self.PTBlock5 = self._make_layer(block=BasicBlock, inplanes=self.dims[3], planes=self.dims[3], num_blocks=2)
        # self.PTBlock6 = self._make_layer(block=BasicBlock, inplanes=self.dims[2], planes=self.dims[2], num_blocks=2)
        # self.PTBlock7 = self._make_layer(block=BasicBlock, inplanes=self.dims[1], planes=self.dims[1], num_blocks=2)

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
        # self.TULayer8 = TULayer(input_a_dim=self.dims[1], input_b_dim = self.dims[0], out_dim=self.dims[0]) # 64 // 2 + 32
        # self.PTBlock8 = PTBlock(in_dim=self.dims[0], hidden_dim=self.dims[0], n_sample=self.neighbor_ks[1])  # 32

        # self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.final_conv = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(self.dims[0], self.final_dim, kernel_size=2, stride=2, dimension=3)
        )
        self.fc = ME.MinkowskiLinear(self.final_dim+self.dims[0], out_channel)

    def forward(self, in_field: ME.TensorField):

        x = in_field

        x0 = self.stem1(x)
        x = self.stem2(x0)
        x1 = self.PTBlock1(x)

        x = self.TDLayer1(x1)
        x2 = self.PTBlock2(x)

        x = self.TDLayer2(x2)
        x3 = self.PTBlock3(x)

        x = self.TDLayer3(x3)
        x4 = self.PTBlock4(x)

        x = self.TULayer5(x4, x3)
        x5 = self.PTBlock5(x)

        x = self.TULayer6(x5, x2)
        x6 = self.PTBlock6(x)

        x = self.TULayer7(x6, x1)
        x7 = self.PTBlock7(x)

        # final big PTBlock
        # x = self.TULayer8(x7, x0)
        # x8, attn_8 = self.PTBlock8(x)
        # x = self.fc(x8)

        x = self.final_conv(x7)
        x = self.fc(me.cat(x0,x))

        # end = time.time()

        #print(f"forward time: {end-start} s")
        # print('PT ratio:{}'.format((pt2 - pt1) / (pt2 - pt0)))

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





