# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function

import MinkowskiEngine as ME

from models.modules.common import ConvType, NormType, get_norm, conv, get_nonlinearity_fn
from models.pct_voxel_utils import separate_batch, voxel2points, points2voxel, PTBlock, cube_query, get_neighbor_feature, pad_zero, make_position_tensor

def separate_batch(coord: torch.Tensor):
    """
        Input:
            coord: (N_voxel, 4) coordinate tensor, coord=b,x,y,z
        Return:
            tensor: (B, N(max n-voxel cur batch), 3), batch index separated
            mask: (B, N), 1->valid, 0->invalid
    """

    # Features donot have batch-ids
    N_voxel = coord.shape[0]
    B = (coord[:,0].max().item() + 1)

    batch_ids = coord[:,0]

    # get the splits of different i_batchA
    splits_at = torch.stack([torch.where(batch_ids == i)[0][-1] for i in torch.unique(batch_ids)]).int() # iter at i_batch_level
    # the returned indices of torch.where is from [0 ~ N-1], but when we use the x[start:end] style indexing, should cover [1:N]
    # example: x[0:1] & x[:1] are the same, contain 1 element, but x[:0] is []
    # example: x[:N] would not raise error but x[N] would

    splits_at = splits_at+1
    splits_at_leftshift_one = splits_at.roll(shifts=1)   # left shift the splits_at
    splits_at_leftshift_one[0] = 0

    len_per_batch = splits_at - splits_at_leftshift_one
    # len_per_batch[0] = len_per_batch[0]+1 # DBEUG: dirty fix since 0~1566 has 1567 values
    N = len_per_batch.max().int()

    assert len_per_batch.sum() == N_voxel

    mask = torch.zeros([B*N], device=coord.device).int()
    new_coord = torch.zeros([B*N, 3], device=coord.device).int() # (B, N, xyz)

    '''
    new_coord: [B,N,3]
    coord-part : [n_voxel, 3]
    idx: [b_voxel, 3]
    '''
    idx_ = torch.cat([torch.arange(len_, device=coord.device)+i*N for i, len_ in enumerate(len_per_batch)])
    idx = idx_.reshape(-1,1).repeat(1,3)
    new_coord.scatter_(dim=0, index=idx, src=coord[:,1:])
    mask.scatter_(dim=0, index=idx_, src=torch.ones_like(idx_, device=idx.device).int())
    mask = mask.reshape([B,N])
    new_coord = new_coord.reshape([B,N,3])

    return new_coord, mask, idx_

def voxel2points_(x_c, x_f_):
    '''
    pack the ME Sparse Tensor feature(batch-dim information within first col of coord)
    [N_voxel_all_batches, dims] -> [bs, max_n_voxel_per_batch, dim]

    idx are used to denote the mask
    '''

    x_c, mask, idx = separate_batch(x_c)
    B = x_c.shape[0]
    N = x_c.shape[1]
    dim = x_f_.shape[1]
    idx_ = idx.reshape(-1,1).repeat(1,dim)
    x_f = torch.zeros(B*N, dim).cuda()
    x_f.scatter_(dim=0, index=idx_, src=x_f_)
    x_f = x_f.reshape([B,N,dim])

    return x_c, x_f, idx

def points2voxel(x, idx):
    '''
    revert the points into voxel's feature
    returns the new feat
    '''
    # the origi_x provides the cooed_map
    B, N, dim = list(x.shape)
    new_x = torch.gather(x.reshape(B*N, dim), dim=0, index=idx.reshape(-1,1).repeat(1,dim))
    return new_x

def gen_pos_enc(x_c, x_f, neighbor, mask, idx_, delta, rel_xyz_only=False, register_map=False):
    k = neighbor.shape[1]
    try:
        relative_xyz = neighbor - x_c[:,None,:].repeat(1,k,1) # (nvoxel, k, bxyz), we later pad it to [B, xyz, nvoxel_batch, k]
    except:
        import ipdb; ipdb.set_trace()
    relative_xyz[:,0,0] = x_c[:,0] # get back the correct batch index, because we messed batch index in the subtraction above
    relative_xyz = pad_zero(relative_xyz, mask) # [B, xyz, nvoxel_batch, k]

    pose_tensor = delta(relative_xyz.float()) # (B, feat_dim, nvoxel_batch, k)
    pose_tensor = make_position_tensor(pose_tensor, mask, idx_, x_c.shape[0]) # (nvoxel, k, feat_dim)S
    # if self.SUBSAMPLE_NEIGHBOR:
        # pose_tensor = pose_tensor[:,self.perms,:]
    # if register_map:
        # self.register_buffer('pos_map', pose_tensor.detach().cpu().data)
    if rel_xyz_only:
        pose_tensor = make_position_tensor(relative_xyz.float(), mask, idx_, x_c.shape[0]) # (nvoxel, k, feat_dim)
    return pose_tensor


def get_sparse_neighbor(k, x, kernel_size=3, stride=1, additional_xf=None):

    if additional_xf is not None:
        x_f = additional_xf
    else:
        x_f = x.F
    N, dim = x_f.shape
    neis = torch.zeros(N,k,dim, device=x_f.device)
    rel_xyz = torch.zeros(N,k,3, device=x.C.device)
    neis_d = x.coordinate_manager.get_kernel_map(x.coordinate_map_key, x.coordinate_map_key,kernel_size=kernel_size, stride=stride)
    for k_ in range(k):
        if not k_ in neis_d.keys():
            continue
        # tmp_neis = torch.zeros(N,dim, device=x.F.device)
        neis_ = torch.gather(x_f, dim=0, index=neis_d[k_][1].reshape(-1,1).repeat(1,dim).long())
        neis[:,k_,:] = torch.scatter(neis[:,k_,:], dim=0, index=neis_d[k_][0].reshape(-1,1).repeat(1,dim).long(), src=neis_)
        rel_xyz_ = torch.gather(x.C[:,1:].float(), dim=0, index=neis_d[k_][1].reshape(-1,1).repeat(1,3).long())
        rel_xyz[:,k_,:] = torch.scatter(rel_xyz[:,k_,:], dim=0, index=neis_d[k_][0].reshape(-1,1).repeat(1,3).long(), src=rel_xyz_)

    # if additional_xf is not None:
        # neis = neis.permute(0,2,1).squeeze(2)

    # N, dim
    return neis, rel_xyz

class apply_choice(Function):
    @staticmethod
    def forward(ctx, out, choice):
        ctx.save_for_backward(out, choice)
        return out*choice

    @staticmethod
    def backward(ctx, g):
        out, choice = ctx.saved_tensors
        g_out = g*torch.ones_like(out, device=out.device) # skip grad of choice on out
        g_choice = (g*out).sum(1).unsqueeze(2)
        return g_out, g_choice

''' ================================ The defined conv ops =================================== '''


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
            if not k_ in neis_d.keys():
                continue
            # tmp_neis = torch.zeros(N,dim, device=x.F.device)
            neis_ = torch.gather(x.F, dim=0, index=neis_d[k_][1].reshape(-1,1).repeat(1,dim).long())
            neis[:,k_,:] = torch.scatter(neis[:,k_,:], dim=0, index=neis_d[k_][0].reshape(-1,1).repeat(1,dim).long(), src=neis_)
            rel_xyz_ = torch.gather(x.C[:,1:].float(), dim=0, index=neis_d[k_][1].reshape(-1,1).repeat(1,3).long())
            rel_xyz[:,k_,:] = torch.scatter(rel_xyz[:,k_,:], dim=0, index=neis_d[k_][0].reshape(-1,1).repeat(1,3).long(), src=rel_xyz_)

        weights = self.linear(rel_xyz).reshape([N, k, self.inplanes, self.planes])
        out_ = (weights*neis.unsqueeze(-1)).sum(dim=1).sum(dim=1) # [N, {K, in_dim,} out_dim]
        out = ME.SparseTensor(features=out_, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        return out

# class TRBlock(PTBlock):
    # # the stride and so on are not actually used, just in coord with the conv interface
    # expansion = 1
    # def __init__(self,
               # inplanes,
               # planes,
               # stride=1,
               # dilation=1,
               # downsample=None,
               # conv_type=ConvType.HYPERCUBE,
               # nonlinearity_type='ReLU',
               # bn_momentum=0.1,
               # D=3):

        # if not inplanes == planes:
            # pass

        # super(TRBlock, self).__init__(
                # in_dim = inplanes,
                # hidden_dim = planes,
                # )


def MinkoskiConvBNReLU(inplanes, planes, kernel_size=1):
    return nn.Sequential(
            ME.MinkowskiConvolution(inplanes, planes, kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(planes),
            ME.MinkowskiReLU(),
            )



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

        # === CodeBook Defs  ===
        self.M = 4 # the codebook size
        self.k = 27
        self.squeeze = 4 # the squeeze dim of the sub-conv

        self.attn_gen_type = "naive"
        assert self.attn_gen_type in ['map_pool', 'cbam','naive','debug']
        if self.attn_gen_type == 'map_pool':
            self.linear_map = nn.Sequential(
                    ME.MinkowskiConvolution(self.inplanes, self.M, kernel_size=1, dimension=3),
                    ME.MinkowskiBatchNorm(self.M)
                    )
            self.avg_pool = nn.Sequential(
                    ME.MinkowskiGlobalAvgPooling(),
                    )
        elif self.attn_gen_type == 'cbam':
            cbam_reduction = 1
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
        elif self.attn_gen_type == 'naive':
            pass
        elif self.attn_gen_type == 'debug':
            pass

        # === Single Conv Definition ===

        self.sample = None

        self.aggregation_type = 'discrete_attn'
        # self.aggregation_type = 'naive_sa'
        assert self.aggregation_type in ['conv_list','cbam','naive_linear', 'naive_sa','debug', 'discrete_attn']

        if self.aggregation_type == 'conv_list':
            self.codebook = nn.ModuleList([])
            for i_ in range(self.M):
                self.codebook.append(
                        nn.Sequential(
                            conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D),
                            ME.MinkowskiBatchNorm(planes),
                            )
                        )
        elif self.aggregation_type == 'cbam':
            self.skip_spatial = True
            pass
        elif self.aggregation_type == 'naive_linear':
            self.conv = nn.Sequential(
                        nn.Conv2d(inplanes, planes, kernel_size=[self.k,1]),
                        nn.BatchNorm2d(planes),
                    )
            self.pos_enc = True
            if self.pos_enc:
                self.gen_pos = nn.Sequential(
                        nn.Conv2d(3, inplanes, 1),
                        nn.BatchNorm2d(inplanes),
                    )
            self.neighbor_type = 'sparse_query'
            self.out_bn_relu = nn.Sequential(
                    ME.MinkowskiBatchNorm(planes),
                    ME.MinkowskiReLU(),
                    )
        elif self.aggregation_type == 'discrete_attn':
            '''
            TODO:
            the main dev here
            within each codebook element is a conv
            the conv weight [dim, K] dot product with the qk,
            to gen the attn_weight for certanin point
            ------------------
            qk_type:
                - conv
                - substration
            conv_v: use conv or linear for gen value
            vec_dim: the attn_map feature dim
            M - codebook size
            temp - the softmax temperature
            '''

            self.M = 4
            # self.M = 1
            self.qk_type = 'conv'
            self.conv_v = False
            self.vec_dim = 1
            # self.vec_dim = planes // 16
            # self.vec_dim = 8
            self.temp = 10
            self.one_hot_choice = False # if one-hot choice, temp is not used
            self.neighbor_type = 'sparse_query'
            self.skip_choice = False # only_used in debug mode
            k = 27

            # self.param_choice = False
            # if self.param_choice:
                # self.choice = nn.Parameter(torch.rand(1,self.M))
            # else:
                # self.gen_choice = nn.Sequential(
                        # nn.Conv2d(inplanes, self.M, kernel_size=[k,1]),
                        # nn.BatchNorm2d(self.M),
                        # )
            if self.inplanes != self.planes:
                self.linear_top = MinkoskiConvBNReLU(inplanes, planes, kernel_size=1)
                # self.downsample = MinkoskiConvBNReLU(inplanes, planes, kernel_size=1)
                self.downsample = ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=3)

            if self.qk_type == 'conv':
                # since conv already contains the neighbor info, so no pos_enc
                self.q = MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=3)
            elif self.qk_type == 'sub':
                self.q = MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=1)
            else:
                raise NotImplementedError

            if self.conv_v == True:
                self.v = MinkoskiConvBNReLU(planes, planes, kernel_size=3)
            else:
                self.v = MinkoskiConvBNReLU(planes, planes, kernel_size=1)

            self.codebook = nn.ModuleList([])
            for i_ in range(self.M):
                self.codebook.append(
                    nn.Sequential(
                        ME.MinkowskiConvolution(planes, planes, kernel_size=3, dimension=3),
                        # ME.MinkowskiChannelwiseConvolution(planes, kernel_size=3, dimension=3),
                        ME.MinkowskiBatchNorm(planes),
                        ME.MinkowskiReLU(),
                        # nn.Conv2d(inplanes, planes, kernel_size=[k,1]),
                        # nn.BatchNorm2d(planes),
                        # nn.ReLU(),
                        )
                    )
            self.out_bn_relu = nn.Sequential(
                    # ME.MinkowskiBatchNorm(planes),
                    ME.MinkowskiReLU(),
                    )

        elif self.aggregation_type == 'naive_sa':
            self.neighbor_type = 'sparse_query'
            self.vec_dim = planes // 4
            self.discrete_qk = True
            if self.discrete_qk:
                self.M = planes // 8
                # for each point [M, dim]
                # choice shape [NPoint, M]
                self.codebook = nn.Parameter(torch.rand(self.M, planes))
                self.gen_choice = nn.Sequential(
                    ME.MinkowskiConvolution(inplanes, self.M, kernel_size=3, dimension=3), # aggregate neighbor feature when gen_choice
                    ME.MinkowskiBatchNorm(self.M),
                    ME.MinkowskiReLU(),
                        )
            self.q = nn.Sequential(
                    ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=3),
                    ME.MinkowskiBatchNorm(planes),
                    ME.MinkowskiReLU(),

                    # nn.Conv2d(inplanes, planes, kernel_size=1),
                    # nn.BatchNorm2d(planes),
                    # nn.ReLU(),
                    )
            self.v = nn.Sequential(
                    ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=3),
                    # ME.MinkowskiConvolution(inplanes, planes, kernel_size=3, dimension=3),
                    ME.MinkowskiBatchNorm(planes),
                    ME.MinkowskiReLU(),

                    ME.MinkowskiConvolution(planes, planes, kernel_size=1, dimension=3),
                    ME.MinkowskiBatchNorm(planes),
                    ME.MinkowskiReLU(),

                    # nn.Conv2d(inplanes, planes, kernel_size=1),
                    # nn.BatchNorm2d(planes),
                    # nn.ReLU(),
                    )
            self.pos_enc = nn.Sequential(
                    # ME.MinkowskiConvolution(3, planes, kernel_size=1, dimension=3),
                    # ME.MinkowskiBatchNorm(planes),
                    # ME.MinkowskiReLU(),
                    nn.Conv2d(3, planes, kernel_size=1),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(),
                    )
            self.attn = nn.Sequential(
                    # ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=3),
                    # ME.MinkowskiBatchNorm(planes),
                    # ME.MinkowskiReLU(),

                    nn.Conv2d(planes, self.vec_dim, kernel_size=1),
                    nn.BatchNorm2d(self.vec_dim),
                    nn.ReLU(),
                    nn.Conv2d(self.vec_dim, self.vec_dim, kernel_size=1),
                    nn.BatchNorm2d(self.vec_dim),
                    nn.ReLU(),

                    )
            self.out_bn_relu = nn.Sequential(
                    # ME.MinkowskiBatchNorm(planes),
                    ME.MinkowskiReLU(),
                    )
        elif self.aggregation_type == 'debug':
            self.test_bn = nn.BatchNorm1d(inplanes)
            self.op = nn.Sequential(
                            conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D),
                            ME.MinkowskiBatchNorm(planes),
                            )


        # === Other Conv Block Defs ===
        self.norm = get_norm(self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = conv(inplanes, planes, kernel_size=1, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
        self.nonlinearity_type = nonlinearity_type

    def schedule_update(self, iter_=None):
        '''
        some schedulable params
        '''
        # self.temp = self.temp*(0.01)**(iter_)
        pass


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
        elif type_ == 'naive':
            attn_weight = None
            # the ops are in the aggregation func
        elif type_ == 'debug':
            attn_weight = None

        # sometimes x will change within this func
        return x, attn_weight

    def aggregation(self, x, attn_weight, type_=None):

        assert type_ is not None

        if type_ == 'conv_list':

            out = None
            for i in range(self.M):

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

        elif 'naive' in type_ or 'attn' in type_:

            npoint, in_dim = tuple(x.F.size())
            if self.neighbor_type == 'sparse_query':

                # gather from sparse_tensor stride_map
                self.stride = 1
                self.kernel_size = 3
                k = self.kernel_size**3

                x_f, neighbor = get_sparse_neighbor(k, x, kernel_size=self.kernel_size, stride=self.stride)

            elif self.neighbor_type == 'knn':
                k = 16
                k = min(k, npoint)

                self.r = 16
                self.USE_KNN = True
                self.cube_query = cube_query(r=self.r, k=k, knn=self.USE_KNN) # make sure it doesnot contain param
                neighbor, mask, idx_ = self.cube_query.get_neighbor(x, x)
                x_f = get_neighbor_feature(neighbor, x)
                x_f_bak = x_f

                if self.pos_enc:
                    pose_tensor = gen_pos_enc(x.C, x_f, neighbor, mask, idx_, delta=self.gen_pos, register_map=False)
                    x_f = x_f + pose_tensor

            x_c, x_f, idx = voxel2points_(x.C, x_f.reshape(npoint, k*in_dim))
            B, npoint_per_batch, _ = x_f.shape

            # x_f = x_f.reshape([B, npoint_per_batch, self.K, in_dim])
            # x_tmp = []
            # for i in range(self.K):
                # x_tmp.append(self.codebook[i](
                    # x_f[:,:,0,:].permute(0,2,1).unsqueeze(-1)
                    # )
                # )
            # out_ = torch.cat(x_tmp, dim=-1).max(-1)[0].permute(0,2,1)

            x_f = x_f.reshape([B, npoint_per_batch, k, in_dim]).permute(0,3,2,1)


            # x_f: [N_voxel, neis_k, dim]
            if type_== 'naive_linear':
                out_ = self.conv(x_f).squeeze(2).permute(0,2,1)
                out_ = points2voxel(out_, idx)

            elif type_ == 'discrete_attn':

                '''
                TODO:
                1st do qk projection: [N, dim, k]
                        - conv: directly use conv neighbor aggregation(extra params), output: [N, vec_dim]
                        - substract: use linear mapping, then gather neighbor & substract. output: [N, vec_dim, k] -> [N, vec_dim] (requires extra transform, memory-hungry)
                2nd: q_ do dot product with M set of conv weights(apply conv): [N, dim, M] -> [N, dim, M], the apply softmax
                    - (q may be vec_dim instead of dim, broadcast to dims)
                3rd: use attn_map: [N, M] to aggregate M convs for each point
                '''


                # res = x.F

                if self.planes != self.inplanes:
                    res = self.downsample(x).F
                    x = self.linear_top(x)
                else:
                    res = x.F


                if self.qk_type == 'conv':
                    q_ = self.q(x)

                v_ = self.v(x)


                # DEBUG: the dim issue, 384 -> 256, if use the output
                # maybe just add a linear_top? to solve all of em
                # making the TR Part same input & same output

                # possible broadcast for filling the vec_dim
                q_f = q_.F.repeat(1, self.planes // self.vec_dim)
                q_= ME.SparseTensor(features=q_f, coordinate_map_key=q_.coordinate_map_key, coordinate_manager=q_.coordinate_manager) # [N, dim]

                # get dot-product of conv-weight & q_

                # DEBUG ONLY
                # v_= ME.SparseTensor(features=torch.ones_like(q_f), coordinate_map_key=q_.coordinate_map_key, coordinate_manager=q_.coordinate_manager) # [N, dim]

                choice = []
                out = []
                for _ in range(self.M):
                    self.codebook[_][0].kernel.requires_grad = False
                    choice_ = self.codebook[_](q_)
                    # DEBUG: whether the grad should attach to the conv weight here?
                    choice.append(choice_.F.reshape(
                        [choice_.shape[0], self.vec_dim, self.planes // self.vec_dim]
                            ).sum(-1)
                        )

                choice = torch.stack(choice, dim=-1)
                if self.M > 1:
                    # apply softmax will result in 0 grad
                    choice = F.softmax(choice / self.temp, dim=-1) # [N, vec_dim, M] 
                else:
                    pass
                choice = choice.repeat(1,self.planes // self.vec_dim,1)
                self.register_buffer('choice_map', choice[:100,:,:])

                # out = choice.sum()
                # x.F.retain_grad()
                # out.backward()
                # print(x.F.grad)
                # print(x.F[0,0])
                # import ipdb; ipdb.set_trace()

                if self.one_hot_choice:
                    N, dim = v_.shape
                    out = torch.zeros([N,dim], device=x.device)
                    for _ in range(self.M):
                        # DEV: split points for different choice
                        # however, if choice has the channle freedom
                        # could not handle
                        assert self.vec_dim == 1 # same point use the same choice 
                        choice_one_hot = torch.argmax(choice, dim=-1)[:,0]  # shape [N]
                        choice_idx = torch.where(choice_one_hot == _)[0]
                        # cur_v_ = v_.features_at_coordinates(v_.C[choice_idx,:].float())
                        if len(choice_idx) > 1:
                            cur_v_ = ME.SparseTensor(
                                    features=v_.F[choice_idx,:],
                                    coordinates=v_.C[choice_idx,:],
                                    coordinate_map_key=x.coordinate_map_key,
                                    coordinate_manager=x.coordinate_manager
                                    )
                            self.codebook[_][0].kernel.requires_grad = True
                            try:
                                cur_out_ = self.codebook[_](cur_v_)
                            except:
                                import ipdb; ipdb.set_trace()
                            out.scatter_(src=cur_out_.F, index=choice_idx.unsqueeze(-1).repeat(1,dim), dim=0)
                        else:
                            pass
                else:
                    out = []
                    for _ in range(self.M):
                            self.codebook[_][0].kernel.requires_grad = True
                            out_ = self.codebook[_](v_)
                            out.append(out_.F)
                    out = torch.stack(out, dim=-1)

                if self.skip_choice:
                    out_ = out.sum(-1)
                else:
                    if self.one_hot_choice:
                        # depreacated, if former is [N,dim,choice]
                        # choice_idx = choice.argmax(-1).unsqueeze(-1)
                        # out_ = torch.gather(input=out, index=choice_idx, dim=2).squeeze(-1)

                        out_ = out
                    else:
                        out_ = (out*choice).sum(-1)

                # '''debug grad'''
                # out = out_.sum()
                # x.F.retain_grad()
                # out.backward()
                # print(self.codebook[0][0].kernel.grad[0])
                # # print(x.F.grad[0][:10])
                # import ipdb; ipdb.set_trace()

                out_ = out_ + res

            elif type_== 'naive_sa':

                if self.discrete_qk:
                    choice = self.gen_choice(x)
                    q_ = torch.matmul(choice.F, self.codebook)
                else:
                    q_ = self.q(x)
                    q_ = q_.F

                # _, x_f0, _= voxel2points(x)
                # x_f0 = x_f0.permute(0,2,1).unsqueeze(2)
                # neis, neighbor = get_sparse_neighbor(k, x, kernel_size=self.kernel_size, stride=self.stride, additional_xf=x_f0)

                v_ = self.v(x)

                q_nei, neighbor = get_sparse_neighbor(k, x, kernel_size=self.kernel_size, stride=self.stride, additional_xf=q_)
                q_ = (q_nei - q_.unsqueeze(1)).permute(2,0,1).unsqueeze(0)
                attn_map = F.softmax(self.attn(q_),dim=-1) # [1, N_voxel, K, dim]
                v_nei, neighbor = get_sparse_neighbor(k, x, kernel_size=self.kernel_size, stride=self.stride, additional_xf=v_.F)
                v_nei = v_nei.permute(2,0,1).unsqueeze(0)

                neighbor_mask = (neighbor.sum(-1)!=0).float().unsqueeze(-1)

                neighbor = (neighbor - x.C[:,1:].unsqueeze(1))*neighbor_mask
                attn_map = attn_map*(neighbor_mask.permute(2,0,1).unsqueeze(0))
                v_nei = v_nei*(neighbor_mask.permute(2,0,1).unsqueeze(0))

                pos_enc = self.pos_enc(neighbor.permute(2,0,1).unsqueeze(0))
                self.register_buffer('pos_map', pos_enc.mean(1))
                self.register_buffer('attn_map', attn_map.mean(1))
                attn_map = attn_map.repeat(1, self.planes // self.vec_dim, 1, 1)
                out_ = (attn_map)*(pos_enc + v_nei)
                out_ = out_.sum(-1).squeeze(0).permute(1,0)

                # if (torch.isnan(q_).sum() > 0):
                    # import ipdb; ipdb.set_trace()

                # num_nonzero = (x_f.sum(1)>0).sum(1) # how many out of 27 is nonzero, div the sum out to avoid large difference

                # out_ = (q_*v_)*(1 / (num_nonzero.unsqueeze(1).unsqueeze(1) + 1e-3)) # [bs, 1, 1, npoint]
                # out_ = self.out_bn_relu(out_).sum(dim=2).permute(0,2,1)

                if (torch.isnan(out_).sum() > 0):
                    import ipdb; ipdb.set_trace()


            '''
            q = self.q(x_f)
            k = self.gen_attn(q - gather_neighbor(q))
            v = gather(selF.v(x_f))

            out = k*v.sum(dim-k)
            '''

            out = ME.SparseTensor(features=out_, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
            out = self.out_bn_relu(out)

        elif type_ == 'debug':
            x_c, x_f, idx = voxel2points(x)
            x_f = self.test_bn(x_f.permute(0,2,1)).permute(0,2,1)
            out = points2voxel(x_f, idx)
            out = ME.SparseTensor(features=out, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
            out = self.op(out)

        return out

    def forward(self, x, iter_=None):
        # K - the code book size 

        out = self.aggregation(x, None, type_=self.aggregation_type)
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
        '''


class SingleConv(nn.Module):
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
    super(SingleConv, self).__init__()

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

  def forward(self, x, iter_=None, aux=None):
    residual = x
    
    out = self.conv(x)
    out = self.norm(out)
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    return out

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

    # if inplanes != planes:
        # self.linear_top = ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=3)

    self.conv = nn.ModuleList([])
    self.M = 4
    for _ in range(self.M):
        if inplanes != planes:
            cur_conv = nn.Sequential(
                    MinkoskiConvBNReLU(inplanes, planes, kernel_size=1),
                    ME.MinkowskiConvolution(planes, planes, kernel_size=1, dimension=3),
                    ME.MinkowskiChannelwiseConvolution(planes, kernel_size=3, dimension=3),
                    )
        else:
            cur_conv = nn.Sequential(
                    MinkoskiConvBNReLU(planes, planes, kernel_size=1),
                    ME.MinkowskiChannelwiseConvolution(planes, kernel_size=3, dimension=3),
                    )

        self.conv.append(
                cur_conv
                )

    self.trainable_weight = False
    if self.trainable_weight:
        self.choice = nn.Parameter(torch.rand([self.M]))

    self.out_bn_relu = nn.Sequential(
            ME.MinkowskiBatchNorm(planes),
            ME.MinkowskiReLU(),
            )

    self.downsample = downsample
    if self.downsample is not None:
        self.downsample = conv(inplanes, planes, kernel_size=1, stride=stride, dilation=dilation, conv_type=conv_type, D=D)

    self.nonlinearity_type = nonlinearity_type

  def forward(self, x, iter_=None):

    if self.inplanes != self.planes:
        residual = self.downsample(x)
        # x = self.linear_top(x)
    else:
        residual = x

    out = []
    for _ in range(self.M):
        out_ = self.conv[_](x)
        if self.trainable_weight:
            out_ = out_*self.choice[_]
        out.append(out_.F)
    out = torch.stack(out, dim=-1).sum(-1)
    out = ME.SparseTensor(features=out, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)

    out = self.out_bn_relu(out)

    out += residual
    out = get_nonlinearity_fn(self.nonlinearity_type, out)

    return out


class SingleChannelConv(nn.Module):
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
    super(SingleChannelConv, self).__init__()

    self.inplanes = inplanes
    self.planes = planes

    # self.conv = ParameterizedConv(
    # self.conv = conv(
        # inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D)
    if inplanes != planes:
        self.conv = nn.Sequential(
                MinkoskiConvBNReLU(inplanes, planes, kernel_size=1),
                ME.MinkowskiConvolution(planes, planes, kernel_size=1, dimension=3),
                ME.MinkowskiChannelwiseConvolution(planes, kernel_size=3, dimension=3),
                )
    else:
        self.conv = nn.Sequential(
                MinkoskiConvBNReLU(planes, planes, kernel_size=1),
                ME.MinkowskiChannelwiseConvolution(planes, kernel_size=3, dimension=3),
                )


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

# class SingleConv(ConvBase):
    # NORM_TYPE = NormType.BATCH_NORM

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

  def forward(self, x, iter_=None, aux=None):
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
