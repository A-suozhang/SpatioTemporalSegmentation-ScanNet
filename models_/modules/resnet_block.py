# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import torch.nn as nn
import torch.nn.functional as F
import torch

import MinkowskiEngine as ME

from models.modules.common import ConvType, NormType, get_norm, conv, get_nonlinearity_fn
from models.pct_voxel_utils import separate_batch, voxel2points, points2voxel, PTBlock


class ParameterizedConv(nn.Module):
    def __init__(self,
               inplanes,
               planes,
               kernel_size,
               stride=1,
               dilation=1,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):
        self.inplanes = inplanes
        self.planes= planes
        self.kernel_size = kernel_size
        self.stride = stride
        super(ParameterizedConv, self).__init__()

        self.linear = nn.Linear(3, self.inplanes*self.planes)

    def forward(self, x, iter_=None):

        # k = self.kernel_size**3
        k = self.kernel_size**3 // 2
        N, dim = x.F.shape
        neis = torch.zeros(N,k,dim, device=x.F.device)
        rel_xyz = torch.zeros(N,k,3, device=x.C.device)
        neis_d = x.coordinate_manager.get_kernel_map(x.coordinate_map_key, x.coordinate_map_key,kernel_size=self.kernel_size, stride=self.stride)
        for k_ in range(k):
            # TODO: possible that no value 
            if not k_ in neis_d.keys():
                continue
            # tmp_neis = torch.zeros(N,dim, device=x.F.device)
            neis_ = torch.gather(x.F, dim=0, index=neis_d[k_][1].reshape(-1,1).repeat(1,dim).long())
            neis[:,k_,:] = torch.scatter(neis[:,k_,:], dim=0, index=neis_d[k_][0].reshape(-1,1).repeat(1,dim).long(), src=neis_)
            rel_xyz_ = torch.gather(x.C[:,1:].float(), dim=0, index=neis_d[k_][1].reshape(-1,1).repeat(1,3).long())
            rel_xyz[:,k_,:] = torch.scatter(rel_xyz[:,k_,:], dim=0, index=neis_d[k_][0].reshape(-1,1).repeat(1,3).long(), src=rel_xyz_)

        # if self.planes != self.inplanes:
            # import ipdb; ipdb.set_trace()

        weights = self.linear(rel_xyz).reshape([N, k, self.inplanes, self.planes])
        out_ = (weights*neis.unsqueeze(-1)).sum(dim=1).sum(dim=1) # [N, {K, in_dim,} out_dim]
        out = ME.SparseTensor(features=out_, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        return out

class TRBlock(PTBlock):
    # the stride and so on are not actually used, just in coord with the conv interface
    expansion = 1
    def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):

        if not inplanes == planes:
            pass
            # import ipdb; ipdb.set_trace()

        super(TRBlock, self).__init__(
                in_dim = inplanes,
                hidden_dim = planes,
                )


# MultiConv
class MultiConv(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM
    def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):

        super(MultiConv, self).__init__()


        self.inplanes = inplanes
        self.planes = planes

        # === CodeBook Defs  ===
        self.K = 4 # the codebook size
        self.squeeze = 4 # the squeeze dim of the sub-conv

        self.attn_gen_type = "cbam"
        assert self.attn_gen_type in ['map_pool', 'cbam']
        if self.attn_gen_type == 'map_pool':
            self.linear_map = nn.Sequential(
                    ME.MinkowskiConvolution(self.inplanes, self.K, kernel_size=1, dimension=3),
                    ME.MinkowskiBatchNorm(self.K)
                    )
            self.avg_pool = nn.Sequential(
                    ME.MinkowskiGlobalAvgPooling(),
                    )
        elif self.attn_gen_type == 'cbam':
            cbam_reduction = 2
            self.single_conv = conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
            self.mlp_c = nn.Sequential(
                    nn.Linear(planes, planes // cbam_reduction),
                    nn.ReLU(),
                    nn.Linear(planes // cbam_reduction, planes),
                    nn.ReLU(),
                    )
            self.conv_s = nn.Sequential(
                    conv(2, 1, kernel_size=5, stride=stride, dilation=dilation, conv_type=conv_type, D=D),
                    ME.MinkowskiBatchNorm(1)
                    )

        # === Single Conv Definition ===

        self.sample = None

        self.aggregation_type = 'cbam'
        assert self.aggregation_type in ['conv_list','cbam']

        if self.aggregation_type == 'conv_list':
            self.codebook = nn.ModuleList([])
            for i_ in range(self.K):
                self.codebook.append(
                        nn.Sequential(
                            conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D),
                            ME.MinkowskiBatchNorm(planes),
                            )
                        )
        elif self.aggregation_type == 'cbam':
            self.skip_spatial = True
            pass

        # === Other Conv Block Defs ===
        self.norm = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = conv(inplanes, planes, kernel_size=1, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
        self.nonlinearity_type = nonlinearity_type

    def gen_attn(self, x, type_=None):

        assert type_ is not None

        if type_ == 'map_pool':

            x_for_attn = self.linear_map(x)  # in_map: [C_in -> K]
            attn_weight = self.avg_pool(x_for_attn)
            # apply softmax on attn_weight
            x_c, x_f, idx = voxel2points(attn_weight)
            x_f = F.softmax(x_f)
            x_f = points2voxel(x_f, idx)
            x_f = F.softmax(x_f)
            attn_weight = ME.SparseTensor(features=x_f, coordinate_map_key=attn_weight.coordinate_map_key, coordinate_manager=attn_weight.coordinate_manager)

        elif type_ == 'cbam':

            x = self.single_conv(x)

            x_c, x_f, idx = voxel2points(x)
            # the channel-wise attn
            x_avg_pool_c = self.mlp_c(x_f.mean(1))  # [B, dim]
            x_max_pool_c = self.mlp_c(x_f.max(1)[0])  # [B, dim]
            # attn_weight_c = F.sigmoid(x_avg_pool_c + x_max_pool_c) # [B,dim]
            attn_weight_c = F.softmax(x_avg_pool_c + x_max_pool_c) # [B,dim]

            # the spatial-wise attn
            if not self.skip_spatial:
                x_avg_pool_s = x_f.mean(-1).unsqueeze(-1)
                x_max_pool_s = x_f.max(-1)[0].unsqueeze(-1)
                x_s = torch.cat([x_avg_pool_s, x_max_pool_s], dim=-1)
                x_s = points2voxel(x_s, idx)
                x_s = ME.SparseTensor(features=x_s, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
                attn_weight_s = self.conv_s(x_s)
                _, attn_weight_s, attn_weight_s_idx = voxel2points(attn_weight_s)
            else:
                # DEBUG: skip attn
                attn_weight_s = None

            self.register_buffer('c_attn_map', attn_weight_c)

            # return a tuple of attn_weight, which is a special case
            attn_weight = attn_weight_c, attn_weight_s

        # sometimes x will change within this func
        return x, attn_weight

    def aggregation(self, x, attn_weight, type_=None):

        assert type_ is not None

        if type_ == 'conv_list':

            out = None
            for i in range(self.K):

                conv_out = self.codebook[i](x)
                x_c, x_f, idx = voxel2points(conv_out)
                attn_weight_ = attn_weight.F[:,i].reshape(-1,1,1)
                out_ = x_f*attn_weight_
                out_ = points2voxel(out_, idx)

                out_ = ME.SparseTensor(features=out_, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
                if out is not None:
                    out = ME.sum(out, out_)
                else:
                    out = out_

        elif type_ == 'cbam':

            # in cbam mode, the conv aggregation happens in the begining of the attn_gen
            attn_weight_c, attn_weight_s = attn_weight
            x_c, x_f, idx = voxel2points(x)

            x_f = x_f*attn_weight_c.unsqueeze(1)
            if not self.skip_spatial:
                x_f = x_f*attn_weight_s   # DEBUG: directly use spatial attn is not suitable, should change form
            else:
                pass
            out = points2voxel(x_f, idx)
            out = ME.SparseTensor(features=out, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)

        return out

    def forward(self, x, iter_=None):
        # K - the code book size 
        '''
        # === debug mode ===
        residual = x
        out = None
        for i in range(self.K):

            out_ = self.codebook[i](x)
            # out_ = ME.SparseTensor(features=out_, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
            if out is not None:
                out = ME.sum(out, out_)
            else:
                out = out_

        # other conv block stuff
        out = self.norm(out)
        if (torch.isnan(out.F).sum() > 0):
            import ipdb; ipdb.set_trace()
        out = get_nonlinearity_fn(self.nonlinearity_type, out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = get_nonlinearity_fn(self.nonlinearity_type, out)

        return out
        '''

        # if want to distinguish train and eval mode
        # use self.norm.training
        # import ipdb; ipdb.set_trace()

        # The Attention Weight Gen
        # weight shape: [1,K]
        residual = x
        if self.downsample is not None:
          residual = self.downsample(x)

        # check nan value
        if (torch.isnan(x.F).sum() > 0):
            import ipdb; ipdb.set_trace()

        x, attn_weight = self.gen_attn(x, type_=self.attn_gen_type)
        if (torch.isnan(x.F).sum() > 0):
            import ipdb; ipdb.set_trace()

        # The Gumbel Sampling:
        # sample from attn_weight distribution instead of directly using W
        pass

        # The codebook building： K set of weights
        # the element: conv / knn & linear
        # Output: the codebook

        out = self.aggregation(x, attn_weight, type_=self.aggregation_type)
        if (torch.isnan(out.F).sum() > 0):
            import ipdb; ipdb.set_trace()


        # other conv block stuff
        out = self.norm(out)
        out = get_nonlinearity_fn(self.nonlinearity_type, out)

        # check nan value
        if (torch.isnan(out.F).sum() > 0):
            import ipdb; ipdb.set_trace()

        out += residual
        out = get_nonlinearity_fn(self.nonlinearity_type, out)

        # check nan value
        if (torch.isnan(out.F).sum() > 0):
            import ipdb; ipdb.set_trace()

        return out


class TestConv(nn.Module):
  expansion = 1
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):
    super(TestConv, self).__init__()

    self.inplanes = inplanes
    self.planes = planes

    # self.conv = ParameterizedConv(
    self.conv = conv(
        inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.norm = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
    self.downsample = downsample
    if self.downsample is not None:
        self.downsample = conv(inplanes, planes, kernel_size=1, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.nonlinearity_type = nonlinearity_type

  def forward(self, x, iter_=None):
    residual = x
    
    out = self.conv(x)
    out = self.norm(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    return out

class ConvBase(nn.Module):
  expansion = 1
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):
    super(ConvBase, self).__init__()

    self.conv = conv(
        inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.norm = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
    self.downsample = downsample
    assert self.downsample is None   # we should not use downsample here
    self.nonlinearity_type = nonlinearity_type

  def forward(self, x, iter_=None):
    residual = x

    out = self.conv(x)
    out = self.norm(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    return out

class SingleConv(ConvBase):
    NORM_TYPE = NormType.BATCH_NORM

class BasicBlockBase(nn.Module):
  expansion = 1
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):
    super(BasicBlockBase, self).__init__()

    self.conv1 = conv(
        inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
    self.conv2 = conv(
        planes,
        planes,
        kernel_size=3,
        stride=1,
        dilation=dilation,
        bias=False,
        conv_type=conv_type,
        D=D)
    self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
    self.downsample = downsample
    self.nonlinearity_type = nonlinearity_type

  def forward(self, x, iter_=None):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    out = self.conv2(out)
    out = self.norm2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    return out


class BasicBlock(BasicBlockBase):
  NORM_TYPE = NormType.BATCH_NORM


class BottleneckBase(nn.Module):
  expansion = 4
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(self,
               inplanes,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               conv_type=ConvType.HYPERCUBE,
               nonlinearity_type='ReLU',
               bn_momentum=0.1,
               D=3):
    super(BottleneckBase, self).__init__()
    self.conv1 = conv(inplanes, planes, kernel_size=1, D=D)
    self.norm1 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)

    self.conv2 = conv(
        planes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    self.norm2 = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)

    self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, D=D)
    self.norm3 = get_norm(self.NORM_TYPE, planes * self.expansion, D, bn_momentum=bn_momentum)

    self.downsample = downsample
    self.nonlinearity_type = nonlinearity_type

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.norm1(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    out = self.conv2(out)
    out = self.norm2(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    out = self.conv3(out)
    out = self.norm3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    return out


class Bottleneck(BottleneckBase):
  NORM_TYPE = NormType.BATCH_NORM
