# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.

from models.resnet import ResNetBase
from models.modules.common import ConvType, NormType, conv, conv_tr, get_norm, get_nonlinearity_fn
from models.modules.resnet_block import BasicBlock, Bottleneck, SingleConv, TestConv, MultiConv, TRBlock

import numpy as np
import MinkowskiEngine.MinkowskiOps as me


class Res16UNetBase(ResNetBase):
  BLOCK = None
  PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
  DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
  INIT_DIM = 32
  OUT_PIXEL_DIST = 1
  NORM_TYPE = NormType.BATCH_NORM
  NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
  # CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE

  # To use the model, must call initialize_coords before forward pass.
  # Once data is processed, call clear to reset the model before calling initialize_coords
  def __init__(self, in_channels, out_channels, config, D=3, **kwargs):
    super(Res16UNetBase, self).__init__(in_channels, out_channels, config, D)

    if not isinstance(self.BLOCK, list): # if single type
        self.BLOCK - [self.BLOCK]*len(self.PLANES)

  def network_initialization(self, in_channels, out_channels, config, D):
    # Setup net_metadata
    dilations = self.DILATIONS
    bn_momentum = config.bn_momentum

    def space_n_time_m(n, m):
      return n if D == 3 else [n, n, n, m]

    if D == 4:
      self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

    if config.xyz_input is not None:
        if config.xyz_input:
            in_channels = in_channels + 3

    # Output of the first conv concated to conv6
    self.inplanes = self.INIT_DIM
    self.conv0p1s1 = conv(
        in_channels,
        self.inplanes,
        kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
        stride=1,
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)

    self.bn0 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

    self.conv1p1s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn1 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

    self.block1 = self._make_layer(
        self.BLOCK[0],
        self.PLANES[0],
        self.LAYERS[0],
        dilation=dilations[0],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)

    self.conv2p2s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn2 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block2 = self._make_layer(
        self.BLOCK[1],
        self.PLANES[1],
        self.LAYERS[1],
        dilation=dilations[1],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)

    self.conv3p4s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn3 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block3 = self._make_layer(
        self.BLOCK[2],
        self.PLANES[2],
        self.LAYERS[2],
        dilation=dilations[2],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)

    self.conv4p8s2 = conv(
        self.inplanes,
        self.inplanes,
        kernel_size=space_n_time_m(2, 1),
        stride=space_n_time_m(2, 1),
        dilation=1,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bn4 = get_norm(self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
    self.block4 = self._make_layer(
        self.BLOCK[3],
        self.PLANES[3],
        self.LAYERS[3],
        dilation=dilations[3],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)
    self.convtr4p16s2 = conv_tr(
        self.inplanes,
        self.PLANES[4],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr4 = get_norm(self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK[4].expansion
    self.block5 = self._make_layer(
        self.BLOCK[4],
        self.PLANES[4],
        self.LAYERS[4],
        dilation=dilations[4],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)
    self.convtr5p8s2 = conv_tr(
        self.inplanes,
        self.PLANES[5],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr5 = get_norm(self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK[5].expansion
    self.block6 = self._make_layer(
        self.BLOCK[5],
        self.PLANES[5],
        self.LAYERS[5],
        dilation=dilations[5],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)
    self.convtr6p4s2 = conv_tr(
        self.inplanes,
        self.PLANES[6],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr6 = get_norm(self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK[6].expansion
    self.block7 = self._make_layer(
        self.BLOCK[6],
        self.PLANES[6],
        self.LAYERS[6],
        dilation=dilations[6],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)
    self.convtr7p2s2 = conv_tr(
        self.inplanes,
        self.PLANES[7],
        kernel_size=space_n_time_m(2, 1),
        upsample_stride=space_n_time_m(2, 1),
        dilation=1,
        bias=False,
        conv_type=self.NON_BLOCK_CONV_TYPE,
        D=D)
    self.bntr7 = get_norm(self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum)

    self.inplanes = self.PLANES[7] + self.INIT_DIM
    self.block8 = self._make_layer(
        self.BLOCK[7],
        self.PLANES[7],
        self.LAYERS[7],
        dilation=dilations[7],
        norm_type=self.NORM_TYPE,
        nonlinearity_type=self.config.nonlinearity,
        bn_momentum=bn_momentum)

    self.final = conv(self.PLANES[7], out_channels, kernel_size=1, stride=1, bias=True, D=D)

  def forward(self, x, save_anchor=False, iter_=None):
    if save_anchor:
        self.anchors = []
    # mapped to transformer.stem1
    out = self.conv0p1s1(x)
    out = self.bn0(out)
    out_p1 = get_nonlinearity_fn(self.config.nonlinearity, out)
    # mapped to transformer.stem2
    out = self.conv1p1s2(out_p1)
    out = self.bn1(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    
    # mapped to transformer.PTBlock1
    out_b1p2 = self.block1(out)
    if save_anchor:
        self.anchors.append(out_b1p2)

    # mapped to transformer.PTBlock2
    out = self.conv2p2s2(out_b1p2)
    out = self.bn2(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out_b2p4 = self.block2(out)
    if save_anchor:
        self.anchors.append(out_b2p4)

    # mapped to transformer.PTBlock3
    out = self.conv3p4s2(out_b2p4)
    out = self.bn3(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out_b3p8 = self.block3(out)
    # if save_anchor:
        # self.anchors.append(out_b3p8)

    # pixel_dist=16
    # mapped to transformer.PTBlock4
    out = self.conv4p8s2(out_b3p8)
    out = self.bn4(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out = self.block4(out)
    # if save_anchor:
        # self.anchors.append(out)


    # pixel_dist=8
    # mapped to transfrormer.PTBlock5
    out = self.convtr4p16s2(out)
    out = self.bntr4(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out = me.cat(out, out_b3p8)
    out = self.block5(out)
    # if save_anchor:
      # self.anchors.append(out)

    # pixel_dist=4
    # mapped to transformer.PTBlock6
    out = self.convtr5p8s2(out)
    out = self.bntr5(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out = me.cat(out, out_b2p4)
    out = self.block6(out)
    if save_anchor:
      self.anchors.append(out)

    # pixel_dist=2
    # mapped to transformer.PTBlock7
    out = self.convtr6p4s2(out)
    out = self.bntr6(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)
    out = me.cat(out, out_b1p2)
    out = self.block7(out)
    if save_anchor:
      self.anchors.append(out)

    # pixel_dist=1
    # mapped to transformer.final_conv
    out = self.convtr7p2s2(out)
    out = self.bntr7(out)
    out = get_nonlinearity_fn(self.config.nonlinearity, out)

    out = me.cat(out, out_p1)
    out = self.block8(out)

    if save_anchor:
        return self.final(out), self.anchors
    else:
        return self.final(out)


class Res16UNet14(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet18(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16UNet34(Res16UNetBase):
  BLOCK = BasicBlock
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

class Res16UNet50(Res16UNetBase):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet101(Res16UNetBase):
  BLOCK = Bottleneck
  LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class Res16UNet14A(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet14A2(Res16UNet14A):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B(Res16UNet14):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14B2(Res16UNet14B):
  LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B3(Res16UNet14B):
  LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16UNet14C(Res16UNet14):
  PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
  PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18D(Res16UNet18):
  PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18E(Res16UNet18):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet18F(Res16UNet18):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34A(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
  PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

class Res16UNetTest(Res16UNetBase):
  # BLOCK = TestConv
  # BLOCK = MultiConv
  BLOCK = [TRBlock]*8
  BLOCK[0] = SingleConv
  LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)

class Res16UNetTestA(Res16UNetTest):
  # BLOCK = [TestConv, TRBlock, TestConv, TRBlock, TestConv, TRBlock, TestConv, TRBlock]
  BLOCK = [TestConv, TestConv, TRBlock, TRBlock, TRBlock, TRBlock, TRBlock, TRBlock]
  LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
  PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
  # PLANES = (8, 16, 32, 64, 64, 32, 24, 24)
  # PLANES = (16, 32, 64, 128, 128, 64, 48, 48)
  # PLANES = (16, 16, 32, 32, 32, 32, 24, 24)
  # PLANES = (16, 16, 16, 16, 16, 16, 24, 24)
  # PLANES = (16, 32, 32, 32, 64, 32, 24, 24)
  # PLANES = (4, 4, 4, 4, 4, 4, 4, 4)





