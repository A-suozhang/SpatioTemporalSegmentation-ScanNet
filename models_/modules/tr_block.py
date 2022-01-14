import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
import numpy as np

import MinkowskiEngine as ME

from models.modules.common import ConvType, NormType, get_norm, conv, get_nonlinearity_fn
from models.modules.resnet_block import *   # dirty and danger op, fix later
from models.pct_voxel_utils import separate_batch, voxel2points, points2voxel, PTBlock, cube_query, get_neighbor_feature, pad_zero, make_position_tensor


POS_ENC_MAP_CUBE_3 = [
                torch.tensor([-1,-1,-1]),
                torch.tensor([0,-1,-1]),
                torch.tensor([1,-1,-1]),
                torch.tensor([-1,0,-1]),
                torch.tensor([0,0,-1]),
                torch.tensor([1,0,-1]),
                torch.tensor([-1,1,-1]),
                torch.tensor([0,1,-1]),
                torch.tensor([1,1,-1]),
                # 
                torch.tensor([-1,-1,0]),
                torch.tensor([0,-1,0]),
                torch.tensor([1,-1,0]),
                torch.tensor([-1,0,0]),
                torch.tensor([0,0,0]),
                torch.tensor([1,0,0]),
                torch.tensor([-1,1,0]),
                torch.tensor([0,1,0]),
                torch.tensor([1,1,0]),
                # 
                torch.tensor([-1,-1,1]),
                torch.tensor([0,-1,1]),
                torch.tensor([1,-1,1]),
                torch.tensor([-1,0,1]),
                torch.tensor([0,0,1]),
                torch.tensor([1,0,1]),
                torch.tensor([-1,1,1]),
                torch.tensor([0,1,1]),
                torch.tensor([1,1,1]),
                ]


POSE_ENC_MAP_CROSS_3 = [
        torch.tensor([0,0,-1]),
        torch.tensor([0,-1,0]),
        torch.tensor([-1,0,0]),
        torch.tensor([0,1,0]),
        torch.tensor([1,0,0]),
        torch.tensor([0,0,1]),
        ]


class GetCodebookWeightStraightThrough(Function):
    @staticmethod
    def forward(ctx, self, neis_l, choice_idx, k_):

        # neis_l: [N, vec_dim]
        # self.codebook[dim, k, M]
        neis_from_codebook = torch.gather(self.codebook[:,k_,:], dim=-1, index=choice_idx.reshape(1,-1).repeat(self.planes,1).long())
        neis_from_codebook = neis_from_codebook.permute(1,0)  # [N. dim]

        ctx.save_for_backward(torch.tensor(self.planes), torch.tensor(self.vec_dim))
        return neis_from_codebook

    @staticmethod
    def backward(ctx, g):
        # input g is the g of codebook_weight
        # straigh through grad
        planes, vec_dim = ctx.saved_tensors

        g_neis_l = g.reshape([-1,vec_dim,planes // vec_dim]).sum(-1)
        return None, g_neis_l, None, None

class GetCodebookWeightStraightThroughQK(Function):
    @staticmethod
    def forward(ctx, self, q_f):

        # q_f: [N, vec_dim]
        # self.codebook[dim, k, M]
        N, dim = q_f.shape
        codebook_weight = []
        choice = []
        diffs = []
        discrete_q = torch.zeros([N, self.planes], device=q_f.device) # discrete-q: [N, dim]
        for m_ in range(self.M):
            codebook_weight_cur_m = self.codebook[:,m_]   # [dim]
            codebook_weight.append(codebook_weight_cur_m)
            choice_cur_m = codebook_weight_cur_m*q_f
            choice_cur_m = self.map_choice(choice_cur_m)  # apply non-lineariity here to prevent grad explosion
            choice_cur_m = choice_cur_m.sum(-1)  # [N]
            choice.append(choice_cur_m)
            discrete_q += self.codebook[None,:,m_]*choice_cur_m[:,None]


        choice = torch.stack(choice, dim=-1) # [N,M]
        self.register_buffer('choice_map', choice)
        # self.register_buffer('coord_map', x.C[:100,:])

        return discrete_q
        # ctx.save_for_backward(torch.tensor(self.planes), torch.tensor(self.vec_dim))

    @staticmethod
    def backward(ctx, g):
        # input g is the g of codebook_weight
        # straigh through grad

        return None, g

class TRBlock(nn.Module):
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

        super(TRBlock, self).__init__()


        self.inplanes = inplanes
        self.planes = planes
        self.h = 1

        # === Single Conv Definition ===
        self.sample = None
        self.type = 'attn'
        self.with_pose_enc = True
        self.with_diverse_reg = False
        # self.type = 'debug'
        if self.with_diverse_reg:
            self.diverse_lambda = 1.

        # self.sparse_kernel = [0,1,2,3,9,10,11,12,15,16,21,22]
        # self.sparse_kernel = [1,3,4,5,7,10,12,13,14,16,19,21,22]
        self.sparse_kernels = [np.arange(27),
                                np.arange(8),
                                np.arange(8),
                                ]
        self.kernel_cfgs = {
                'kernel_size': [3,2,2],
                'dilation':[1,2,3],
                }
        self.k =  sum([len(sp_kernel) for sp_kernel in self.sparse_kernels])  # total K

        assert self.type in ['attn','debug']

        if self.type == 'attn':

            self.vec_dim = 4
            # self.vec_dim = 1
            if self.with_pose_enc:
                self.pos_enc = nn.Sequential(
                    MinkoskiConvBNReLU(3, planes, kernel_size=1),
                    MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=1),
                        )

            # self.vq = True # use vector quantization 
            # self.vq_loss_commit_beta = 1.
            # self.vq_size = 1
            # self.vq_lambda = 0.
            # self.vq_loss = None

            # self.codebook = nn.Parameter(
                    # # torch.nn.init.xavier_uniform_(
                    # torch.nn.init.kaiming_uniform_(
                        # torch.empty(self.planes, self.k, self.vq_size), nonlinearity='relu'
                    # ) / 10 # / 10  to make the data around 1e-2, which is important for training!
                # )

            # self.debug_channel_conv = ME.MinkowskiChannelwiseConvolution(self.planes, kernel_size=3, dimension=3)
            if self.inplanes != self.planes:
                self.linear_top = MinkoskiConvBNReLU(inplanes, planes, kernel_size=1)
                self.downsample = ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=3)

            self.q = nn.Sequential(
                MinkoskiConvBNReLU(planes, planes, kernel_size=3),
                MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=3),
            )
            self.v = MinkoskiConvBNReLU(planes, planes, kernel_size=3)
            self.map_qk = nn.Sequential(
                nn.Linear(self.vec_dim, self.planes),
                nn.ReLU(),
                nn.Linear(self.planes, self.vec_dim),
            )
            self.out_bn_relu = nn.Sequential(
                    ME.MinkowskiBatchNorm(planes),
                    ME.MinkowskiReLU(),
                    )

        elif self.type== 'debug':
            self.op = nn.Sequential(
                            conv(inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, conv_type=conv_type, D=D),
                            ME.MinkowskiBatchNorm(planes),
                            )

    def schedule_update(self, iter_=None):
        '''
        some schedulable params
        '''
        pass

    def get_vq_loss(self, neis_l):
        """
        debug
        """
        pass

    def expand_vec_dim(self,x):
        # x shold be like [N, vec_dim]; [N, vec_dim, M]
        # expand em as [N, dim]; [N, dim, M]
        assert x.shape[1] == self.vec_dim
        if len(x.shape) == 2:
            N, dim = x.shape
            x = x.unsqueeze(2).expand(-1,-1,self.planes*self.h//self.vec_dim).reshape(-1,self.planes*self.h)
        elif len(x.shape) == 3:
            N, dim, M = x.shape
            x = x.unsqueeze(2).expand(-1,-1,self.planes*self.h//self.vec_dim, -1).reshape(-1,self.planes*self.h,M)

        return x

    def get_diversity_reg(self, x):
        # x: [N, vec_dim, K]
        x_ = x.mean(0).mean(0) # [K]
        self.diverse_loss = ((torch.matmul(x_, x_.T) - torch.eye(self.k, device=x.device))**2).sum()
        self.diverse_loss = self.diverse_lambda*self.diverse_loss

    def forward(self, x, iter_=None, aux=None):

        self.schedule_update(iter_)

        if self.type == 'attn':

            if self.planes != self.inplanes:
                res = self.downsample(x)
                x = self.linear_top(x)
            else:
                res = x

            v_ = self.v(x)
            q_ = self.q(x)

            # q_f = q_.F.repeat(1, self.planes // self.vec_dim)
            q_f = q_.F
            q_= ME.SparseTensor(features=q_f, coordinate_map_key=q_.coordinate_map_key, coordinate_manager=q_.coordinate_manager) # [N, dim]

            neis_ds = []
            for _ in range(len(self.sparse_kernels)):
                neis_ds.append(q_.coordinate_manager.get_kernel_map(q_.coordinate_map_key,
                                                            q_.coordinate_map_key,
                                                            kernel_size=self.kernel_cfgs['kernel_size'][_],
                                                            dilation=self.kernel_cfgs['dilation'][_],
                                                            stride=1,
                                                                )
                                                            )
            # q_.coordinate_manager == v_.coordinate_manager 
            N, dim = v_.F.shape
            neis_l = []
            sparse_masks = []
            for i_sk, sparse_kernel in enumerate(self.sparse_kernels):
                neis_d = neis_ds[i_sk]
                sparse_mask = []
                sparse_kernel_ = list(set(neis_d.keys()).intersection(set(sparse_kernel)))   # get the intersection of current neis and sparse-mask
                for k_ in sparse_kernel_:

                    # if not k_ in neis_d.keys():
                        # continue

                    neis_sparse_mask_ = torch.gather(x.F, dim=0, index=neis_d[k_][0].reshape(-1,1).expand(-1,self.planes).long())
                    neis_sparse_mask = torch.zeros(N,self.planes, device=q_.F.device)  # DEBUG: not sure if needs decalre every time
                    neis_sparse_mask = torch.scatter(neis_sparse_mask, dim=0, index=neis_d[k_][1].reshape(-1,1).expand(-1,self.planes).long(), src=neis_sparse_mask_)
                    sparse_mask_cur_k = (neis_sparse_mask.abs().sum(-1) > 0).float()
                    sparse_mask.append(sparse_mask_cur_k)
                sparse_masks.append(sparse_mask)

                # for k_ in range(self.k):
                for i_k, k_ in enumerate(sparse_kernel_):

                    # if not k_ in neis_d.keys():
                        # continue

                    neis_ = torch.gather(q_.F, dim=0, index=neis_d[k_][0].reshape(-1,1).expand(-1,self.vec_dim).long())
                    neis = torch.zeros(N,self.vec_dim, device=q_.F.device)  # DEBUG: not sure if needs decalre every time
                    neis = torch.scatter(neis, dim=0, index=neis_d[k_][1].reshape(-1,1).expand(-1,self.vec_dim).long(), src=neis_)
                    try:
                        sparse_mask_cur_k = sparse_mask[i_k]
                    except IndexError:
                        import ipdb; ipdb.set_trace()
                    neis = neis - (q_.F*sparse_mask_cur_k.unsqueeze(-1).expand(-1, self.vec_dim))

                    neis = self.map_qk(neis)  # apply a linear layer over the neighbor
                    neis = neis*sparse_mask_cur_k.unsqueeze(-1).expand(-1,self.vec_dim)

                    neis_l.append(neis)


            neis_l = torch.stack(neis_l, dim=-1) # [N, vec_dim, K]
            neis_l = F.softmax(neis_l, dim=-1)

            if self.with_diverse_reg:
                self.get_diversity_reg(neis_l)

            out = torch.zeros([N, dim], device=q_.F.device)
            if self.with_pose_enc:
                pose_enc2save = []

            # for k_ in range(self.k):
            for i_sk, sparse_kernel in enumerate(self.sparse_kernels):
                neis_d = neis_ds[i_sk]
                sparse_kernel_ = list(set(neis_d.keys()).intersection(set(sparse_kernel)))   # get the intersection of current neis and sparse-mask
                for i_k, k_ in enumerate(sparse_kernel_):

                    if not k_ in neis_d.keys():
                        continue

                    if self.with_pose_enc:
                        x_c = ME.SparseTensor(features=v_.C[:,1:].float(), coordinate_map_key=v_.coordinate_map_key, coordinate_manager=v_.coordinate_manager)
                        pos_enc = self.pos_enc(x_c)
                        pos_enc = self.expand_vec_dim(pos_enc.F)
                        v_f = v_.F + pos_enc
                        pose_enc2save.append(pos_enc.mean(-1))
                    else:
                        v_f = v_.F

                    neis_v_ = torch.gather(v_f, dim=0, index=neis_d[k_][0].reshape(-1,1).expand(-1,dim).long())
                    neis_v = torch.zeros(N,dim, device=q_.F.device)  # DEBUG: not sure if needs decalre every time
                    neis_v = torch.scatter(neis_v, dim=0, index=neis_d[k_][1].reshape(-1,1).expand(-1,dim).long(), src=neis_v_)
                    sparse_mask_cur_k_v = sparse_masks[i_sk][i_k]

                    neis_v = neis_v*sparse_mask_cur_k_v.unsqueeze(-1).expand(N, self.planes)   # [N, dim]
                    out += neis_v*neis_l[:,:,i_k].unsqueeze(-1).expand(-1,-1,self.planes//self.vec_dim).reshape(-1,self.planes) # [N. dims]

            if self.with_pose_enc:
                pose_enc2save = torch.stack(pose_enc2save)
                self.register_buffer("pos_enc_map", pose_enc2save)

            out = ME.SparseTensor(features=out, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
            out = self.out_bn_relu(out)
            out = out + res
            return out

        elif self.type == 'debug':
            # x_c, x_f, idx = voxel2points(x)
            # x_f = self.test_bn(x_f.permute(0,2,1)).permute(0,2,1)
            # out = points2voxel(x_f, idx)
            # out = ME.SparseTensor(features=out, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
            # out = self.op(out)
            out = self.op(x)
        else:
            raise NotImplementedError

        return out

class DiscreteAttnTRBlock(nn.Module): # ddp could not contain unused parameter, so donnot inherit from TRBlock
    expansion=1
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

        super(DiscreteAttnTRBlock, self).__init__()


        self.inplanes = inplanes
        self.planes = planes
        self.k = 27

        # === Single Conv Definition ===
        self.sample = None

        self.type = 'discrete_attn'
        assert self.type in ['discrete_attn']

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

        self.h = 1 # the num-head, noted that since all heads are parallel, could view as expansion
        self.M = 3
        self.qk_type = 'pairwise'
        # self.qk_type = 'conv'
        self.conv_v = True
        # self.vec_dim = 1
        # self.vec_dim = 4
        self.vec_dim = self.planes // 8
        self.top_k_choice = False
        # self.neighbor_type = 'sparse_query'
        self.k = 27
        self.temp_ = 1.e0 # the initial temp
        # self.temp_ = 1.e3 # the initial temp

        # === some additonal tricks ===
        self.skip_choice = False # only_used in debug mode, notice that this mode contains unused params, so could not support ddp for now
        self.gradual_split = False
        self.smooth_choice = False

        self.diverse_reg = False
        self.diverse_lambda = (1.e-4)

        self.codebook_prior = False
        self.hard_mask = False

        self.sparse_pattern_reg = False

        num_class = 21
        self.with_label_embedding = False
        if self.with_label_embedding:
            self.label_reg_lambda = (5.e-5)
            self.aux_head = ME.MinkowskiConvolution(self.planes, num_class, kernel_size=1, dimension=3)
            self.aux_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        if self.with_label_embedding:
            self.label_embedding = nn.Parameter(torch.rand(num_class, planes))

        if self.inplanes != self.planes:
            self.linear_top = MinkoskiConvBNReLU(inplanes, planes, kernel_size=1)
            self.downsample = ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=3)

        if self.conv_v == True:
            self.v = nn.Sequential(
                    MinkoskiConvBNReLU(planes, planes, kernel_size=3),
                    MinkoskiConvBNReLU(planes, planes*self.h, kernel_size=1),
                    )
        else:
            self.v = MinkoskiConvBNReLU(planes, planes*self.h, kernel_size=1)

        self.codebook = nn.ModuleList([])
        self.CUSTOM_KERNEL = True
        if self.CUSTOM_KERNEL:
            kgargs0 = {
                "kernel_size": 3,
                "stride": 1,
                "dilation": 2,
                "region_type":ME.RegionType.HYPER_CROSS,
                # "region_type":ME.RegionType.HYPER_CUBE,
                # "region_type": ME.RegionType.CUSTOM,
                # "region_offsets": ro0,
                "dimension": 3,
                }
            kgargs1 = {
                "kernel_size": 3,
                # "kernel_size": 2,    # DEBUG_ONLY!
                "stride": 1,
                "dilation": 1,
                "region_type":ME.RegionType.HYPER_CUBE,
                # "region_type": ME.RegionType.CUSTOM,
                # "region_offsets": ro0,
                "dimension": 3,
                }
            kgargs2 = {
                "kernel_size": 3,
                "stride": 1,
                "dilation": 3,
                "region_type":ME.RegionType.HYPER_CROSS,
                # "region_type": ME.RegionType.CUSTOM,
                # "region_offsets": ro0,
                "dimension": 3,
                }
            self.kgargs = [kgargs0, kgargs1, kgargs2]
            # self.kgargs = [kgargs0, kgargs1]
            kgs = [ME.KernelGenerator(**kg) for kg in self.kgargs]
            for i_ in range(self.M):
                self.codebook.append(
                    nn.Sequential(
                        ME.MinkowskiChannelwiseConvolution(planes*self.h, kernel_size=3, dimension=3, kernel_generator=kgs[i_]),
                        # ME.MinkowskiBatchNorm(planes*self.h),
                        # ME.MinkowskiReLU(),
                        )
                    )
            # DEBUG:  rewrite Q for custom-kernel
            if not self.skip_choice:
                if self.qk_type == 'conv':
                    self.q = nn.ModuleList([])
                    for i_ in range(self.M):
                        self.q.append(
                            nn.Sequential(
                                ME.MinkowskiConvolution(planes,self.vec_dim,  kernel_size=3, dimension=3, kernel_generator=kgs[i_]),
                                )
                            )
                elif self.qk_type == 'pairwise':
                    self.q = MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=1)
                    # self.pos_enc = MinkoskiConvBNReLU(3, self.vec_dim, kernel_size=1)

        else:
            kgargs0 = {
                "kernel_size": 3,
                "stride": 1,
                "dilation": 1,
                "region_type":ME.RegionType.HYPER_CUBE,
                # "region_type": ME.RegionType.CUSTOM,
                # "region_offsets": ro0,
                "dimension": 3,
                }
            self.kgargs = [kgargs0]*self.M
            for i_ in range(self.M):
                self.codebook.append(
                    nn.Sequential(
                        # ME.MinkowskiConvolution(planes*self.h, planes*self.h, kernel_size=3, dimension=3),
                        ME.MinkowskiChannelwiseConvolution(planes*self.h, kernel_size=3, dimension=3),
                        # ME.MinkowskiBatchNorm(planes),
                        # ME.MinkowskiReLU(),
                        )
                    )
            if not self.skip_choice:
                if self.qk_type == 'conv':
                    # since conv already contains the neighbor info, so no pos_enc
                    self.q = nn.Sequential(
                        ME.MinkowskiConvolution(planes, self.vec_dim, kernel_size=3,dimension=3),
                        # ME.MinkowskiBatchNorm(self.vec_dim),
                            )
                    # self.q = MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=3)
                    # self.q = MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=1) # DEBUG_OBLY!
                elif self.qk_type == 'pairwise':
                    self.q = MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=1)
                    # self.pos_enc = MinkoskiConvBNReLU(3, self.vec_dim, kernel_size=1)
                    # self.map_qk = nn.Linear(self.vec_dim, self.vec_dim)
                else:
                    raise NotImplementedError

        # specify some of the Priors
        if self.codebook_prior:

            mask0 = []
            mask1 = np.array([
                    [10,11,12,20,21,22],
                    [1,2,3,4,5,6,10,21,20],
                    [3,4,5,6,7,8,9,10,11],
                    [17,18,19,20,22,23,24],
                    ])
            mask2 = []
            self.codebook_masks = [mask0, mask1, mask2]
            # for _ in range(len(self.codebook))
                # mask_ = mask_empty.clone()
                # mask_[:10,:].fill_(1)
                # masks.append(mask_)

            for _ in range(len(self.codebook)):
                # mask_empty = torch.zeros_like(self.codebook[0][0].kernel)
                new_kernel = self.codebook[_][0].kernel
                k_, dim_ = new_kernel.shape
                if len(self.codebook_masks[_])>0:
                    assert self.vec_dim % len(self.codebook_masks[_]) == 0
                if len(self.codebook_masks[_])>1:
                    dim_per_mask = dim_ // len(self.codebook_masks[_])
                else:
                    dim_per_mask = dim_
                for m_ in range(len(self.codebook_masks[_])):
                    new_kernel[self.codebook_masks[_][m_],dim_per_mask*m_:dim_per_mask*(m_+1)] = 0
                # if _ == 0:
                    # new_kernel = new_kernel[:9,:].fill_(new_kernel.max())
                # elif _ == 1:
                    # new_kernel = new_kernel[10:18].fill_(new_kernel.max())
                self.codebook[_][0].kernel = nn.Parameter(self.codebook[_][0].kernel)

            # codebook_weight = torch.stack([m[0].kernel for m in self.codebook])
            # torch.save(codebook_weight, '/home/zhaotianchen/project/point-transformer/SpatioTemporalSegmentation-ScanNet/plot/codebook_weight.pth')
        self.out_bn_relu = nn.Sequential(
                ME.MinkowskiConvolution(planes*self.h, planes, kernel_size=1, dimension=3),
                ME.MinkowskiBatchNorm(planes),
                ME.MinkowskiReLU(),
                )

        if self.smooth_choice:
            self.smooth_conv = ME.MinkowskiChannelwiseConvolution(self.M, kernel_size=3, dimension=3) # the fixed low-pass filter for local neighbors
            dummy_kernel = nn.Parameter(torch.ones_like(self.smooth_conv.kernel))
            self.smooth_conv.kernel = dummy_kernel
            self.smooth_conv.kernel.requires_grad = False

    def expand_vec_dim(self,x):
        # x shold be like [N, vec_dim]; [N, vec_dim, M]
        # expand em as [N, dim]; [N, dim, M]
        assert x.shape[1] == self.vec_dim
        if len(x.shape) == 2:
            N, dim = x.shape
            x = x.unsqueeze(2).expand(-1,-1,self.planes*self.h//self.vec_dim).reshape(-1,self.planes*self.h)
        elif len(x.shape) == 3:
            N, dim, M = x.shape
            x = x.unsqueeze(2).expand(-1,-1,self.planes*self.h//self.vec_dim, -1).reshape(-1,self.planes*self.h,M)

        return x

    def get_sparse_pattern(self, x, type_=2):

        # FORMULA 1: get codebook kernel shapes and directly use the sparse-pattern matching 
        # as the guidance of choice

        # PROBLEM 1: how to support more flexible density estimation
        #    - currently only support matching kernel & its neighbor
        #    - the very sparse scenarios are hard to distinguish(3x3 kernel is too small, all have 0 neis)
        #    - memory bottleneck needs benchmarking, are neis_d itself very mem-consuming?
        if type_ == 1:

            sparse_patterns= []  # [M]
            for m_ in range(self.M):
                kgargs = self.kgargs[m_]
                if 'dimension' in kgargs.keys():
                    del kgargs['dimension']
                neis_d = x.coordinate_manager.get_kernel_map(x.coordinate_map_key,
                                                                    x.coordinate_map_key,
                                                                    **kgargs
                                                                    )
                N = x.C.shape[0]
                # its easy to get how many matched elements of cur-point & kernel
                # but the kernel shape is hard to be flexible, like i need to index the lower-right part
                if self.codebook_prior:
                    # only when codebook-prior is given, each point would have different pattern
                    sparse_pattern_ = torch.zeros([N, self.vec_dim], device=x.device)
                else:
                    sparse_pattern_ = torch.zeros([N, 1], device=x.device)

                if hasattr(self, "codebook_masks"):
                    cur_mask = self.codebook_masks[m_]
                else:
                    cur_mask = []

                cur_k = len(neis_d.keys())
                for k_ in range(cur_k):

                    if not k_ in neis_d.keys():
                            continue

                    if len(cur_mask)>0:
                        for i_ in range(len(cur_mask)):
                            if k_ in cur_mask[i_]:  # for masked k
                                continue
                            else:
                                sparse_pattern_[neis_d[k_][0].long(),i_] +=1
                    else:
                        sparse_pattern_[neis_d[k_][0].long(),:] +=1

                if len(cur_mask)>0:
                    for i_ in range(len(cur_mask)):
                        sparse_pattern_[:,i_] = sparse_pattern_[:,i_] / (cur_k - len(cur_mask[i_]))
                else:
                    sparse_pattern_ = sparse_pattern_ / cur_k
                sparse_patterns.append(sparse_pattern_)
            sparse_patterns = torch.stack(sparse_patterns, dim=-1)

            self.register_buffer("sparse_patterns",sparse_patterns)

            # Reg Type1:  encourage the kernel to lean to map with more matching neighbors
            temp_ = 0.1
            eps = 1.e-3
            self.sparse_pattern = F.softmax((sparse_patterns+eps)/temp_, dim=-1)  # [N. M]

        # formula 2: MultiScale Estimation of how sparse a point is 
        # apply softmax in the normalized N points dimension
        # calc the relative sparsity distance to many centers as regs
        elif type_ == 2:

            eps = 1.e-3
            T = 25

            neis_d = x.coordinate_manager.get_kernel_map(
                                                        x.coordinate_map_key,
                                                        x.coordinate_map_key,
                                                        kernel_size=3,
                                                        stride=1,
                                                        )
            N = x.C.shape[0]
            sparse_pattern_ = torch.zeros([N, 1], device=x.device)
            for k_ in range(len(neis_d)):
                if not k_ in neis_d.keys():
                    continue
                else:
                    sparse_pattern_[neis_d[k_][0].long(),:] +=1
            sparse_pattern_ = sparse_pattern_ / sparse_pattern_.max()
            codebook_centers = torch.arange(0,1,1/self.M,device=x.device)
            self.sparse_patterns = F.softmax((1/(sparse_pattern_ - codebook_centers + eps).abs())/T, dim=-1).unsqueeze(1) # [N,1, M]

    def get_vq_loss(self, neis_l):
        pass

        # N,_,_ = neis_l.shape
        # diff1 = []
        # diff2 = []
        # for i_ in range(self.vq_size):
            # # the below method will cause burst in memory
            # diff1_ = ((neis_l.unsqueeze(2).expand(-1,-1,self.planes//self.vec_dim,-1).reshape([N,self.planes,self.k]).detach() - self.codebook[:,:,i_].unsqueeze(0)).abs()).sum()
            # diff2_ = ((neis_l.unsqueeze(2).expand(-1,-1,self.planes//self.vec_dim,-1).reshape([N,self.planes,self.k]) - self.codebook[:,:,i_].unsqueeze(0).detach()).abs()).sum()

            # diff1.append(diff1_)
            # diff2.append(diff2_)
        # diff1 = torch.stack(diff1, dim=-1).sum()
        # diff2 = torch.stack(diff2, dim=-1).sum()
        # self.vq_loss = (diff1 + self.vq_loss_commit_beta*diff2 ) / N # normalize by N points
        # self.vq_loss = self.vq_loss*self.vq_lambda # apply vq_lambda

    def get_diversity_reg(self):
        self.diverse_loss = 0.
        codebook_weight = torch.stack([torch.sqrt(F.softmax(self.codebook[_][0].kernel, dim=0)) for _ in range(self.M)])   # apply sqrt on the reg term
        for m_ in range(self.M):
            for m_2 in range(self.M):
                if m_2 < m_:
                    self.diverse_loss += ((torch.matmul(codebook_weight[m_], codebook_weight[m_2].T) - torch.eye(self.k, device=codebook_weight.device))**2).sum()

        self.diverse_loss = self.diverse_lambda*self.diverse_loss

    def get_label_embedding(self, x, aux):
        if self.with_label_embedding:
            self.label_reg = 0.
            if aux is not None:
                aux = aux.features_at_coordinates(x.C.float())
                aux_pred = self.aux_head(x)
                self.label_reg = self.aux_criterion(aux_pred.F, aux.squeeze(-1).long())
                # aux_ = torch.gather(self.label_embedding, dim=0, index=(aux+1).long().expand(-1,self.planes)) # to avoid -1
                # self.label_reg = ((x.F.mean(-1))*(aux_.mean(-1))).sum()
                # self.label_reg = self.label_reg*self.label_reg_lambda
            else:
                self.label_reg = 0.

    def schedule_update(self, iter_=None):
        '''
        some schedulable params
        '''
        # ======= the temp annealing for choice =============
        self.temp = (self.temp_)**(1-iter_) # start from the temp, end with 0

        if self.skip_choice == True and iter_> 0.1:
            self.skip_choice = False
            print('SkipChoice Warmup Done, Start training choice qk')

        if self.skip_choice == False and not hasattr(self, "q"):
            self.q = nn.Sequential(
                ME.MinkowskiConvolution(self.planes, self.vec_dim, kernel_size=3,dimension=3),
                ME.MinkowskiBatchNorm(self.vec_dim),
                    )
            self.q.to(self.codebook[0][0].kernel.device)

        pass

        # ========== Split Codebook ==============

        # split_at = [0.1, 0.2, 0.3]
        # split_done = [False]*3

        # if hasattr(self, 'gradual_split'):
            # # for i_ in range(len(split_at)):
                # # if iter_ > split_at[i_] and split_done[i_] is False:
                    # # self.gradual_split = True
                    # # split_done[i_] = True

            # # copy the codebook
            # if self.gradual_split:
                # for _ in range(self.M):
                    # new_conv = nn.Sequential(
                        # ME.MinkowskiChannelwiseConvolution(self.planes, kernel_size=3, dimension=3),
                    # )
                    # new_conv[0].kernel = nn.Parameter(self.codebook[_][0].kernel)
                    # self.codebook.append(new_conv)
                # self.M = self.M*2
                # self.gradual_split  = False

        # # ====== Begin At ============

        # begin_qk_at = 0.3
        # if iter_ < begin_qk_at:
            # self.skip_choice = True
        # else:
            # self.skip_choice = False

        # ========= Temperature Annealing ==============

        # if not hasattr(self, 'temp0'):
            # self.temp0 = self.temp

        # self.temp = self.temp0*(0.01)**(iter_)

    def forward(self, x, iter_=None, aux=None):
        '''
        TODO
        1st do qk projection: [N, dim, k]
                - conv: directly use conv neighbor aggregation(extra params), output: [N, vec_dim]
                - substract: use linear mapping, then gather neighbor & substract. output: [N, vec_dim, k] -> [N, vec_dim] (requires extra transform, memory-hungry)
        2nd: q_ do dot product with M set of conv weights(apply conv): [N, dim, M] -> [N, dim, M], the apply softmax
            - (q may be vec_dim instead of dim, broadcast to dims)
        3rd: use attn_map: [N, M] to aggregate M convs for each point
        '''
        self.register_buffer('coord_map', x.C)
        self.schedule_update(iter_)

        if self.sparse_pattern_reg:
            self.get_sparse_pattern(x)

        if self.planes != self.inplanes:
            res = self.downsample(x)
            x = self.linear_top(x)
        else:
            res = x

        # ===== some code for saving the coord-data and calc the sprase "hit-rate"
        # d = {}
        # neis_d = x.coordinate_manager.get_kernel_map(x.coordinate_map_key,
                                                            # x.coordinate_map_key,
                                                            # kernel_size=3,
                                                            # stride=1,
                                                            # )
        # d['all_c'] = x.C[:,1:]
        # N = x.C.shape[0]
        # all_neis = []
        # for k_ in range(self.k):

            # if not k_ in neis_d.keys():
                    # continue

            # neis_ = torch.gather(x.C[:,1:].float(), dim=0, index=neis_d[k_][0].reshape(-1,1).repeat(1,3).long())
            # neis = torch.zeros(N,3, device=x.F.device)
            # neis.scatter_(dim=0, index=neis_d[k_][1].reshape(-1,1).repeat(1,3).long(), src=neis_)
            # all_neis.append(neis)
        # all_neis = torch.stack(all_neis)

        # d['neis'] = all_neis

        # aux_  = aux.features_at_coordinates(x.C.float())
        # aux_c =  aux.coordinate_manager.get_coordinates(x.coordinate_map_key).float()
        # aux_f  = aux.features_at_coordinates(aux_c)

        # d['aux_c'] = aux_c
        # d['aux_f'] = aux_f

        # d['aux_c'] = aux.C
        # d['aux_f'] = aux.F
        # d['label'] = aux_f.squeeze(-1)

        # torch.save(d, '/home/zhaotianchen/project/point-transformer/SpatioTemporalSegmentation-ScanNet/plot/final/sparse_extent_layer1.pth')
        # import ipdb; ipdb.set_trace()

        # =====================================================================================================================

        v_ = self.v(x)

        if self.diverse_reg:
            self.get_diversity_reg()

        if not self.skip_choice:

            if self.qk_type == 'conv':
                if not self.CUSTOM_KERNEL:
                    q_ = self.q(x)
                    q_f = self.expand_vec_dim(q_.F)
                    q_= ME.SparseTensor(features=q_f, coordinate_map_key=q_.coordinate_map_key, coordinate_manager=q_.coordinate_manager) # [N, dim]
                    N, dim = q_.F.shape
                    qs = [q_]*self.M
                else:
                    qs = []
                    for _ in range(self.M):
                        q_ = self.q[_](x)
                        q_f =self.expand_vec_dim(q_.F)
                        qs.append(
                            ME.SparseTensor(features=q_f, coordinate_map_key=q_.coordinate_map_key, coordinate_manager=q_.coordinate_manager) # [N, dim]
                                )
                        N, dim = q_f.shape

                # get dot-product of conv-weight & q_
                choice = []
                out = []
                for _ in range(self.M):
                    self.codebook[_][0].kernel.requires_grad = False
                    choice_ = self.codebook[_](qs[_])
                    choice.append(choice_.F.reshape(
                        [choice_.shape[0], self.vec_dim, self.planes*self.h // self.vec_dim]
                            ).sum(-1)
                        )
                choice = torch.stack(choice, dim=-1)
                eps = 1.e-3
                # self.temp = 0.1
                # choice = choice.reshape([N,self.M*self.vec_dim])

                # debug: actually to show the effect of the sparse-pattern reg, we should multuiply it afterwards, however, too strong aug seems to bring lossy perf
                # needs checking
                if self.M > 1: # if M==1, skip softmax since there is only 1 value
                    choice = F.softmax((choice)/self.temp, dim=-1) # [N, vec_dim, M] 
                    # choice = F.softmax((choice+eps)/self.temp, dim=-1) # [N, vec_dim, M] 
                    # choice = choice.reshape([N, self.vec_dim, self.M])
                else:
                    pass

                if self.sparse_pattern_reg:
                    choice = choice*self.sparse_patterns

                # attn_map = torch.stack([self.codebook[_][0].kernel for _ in range(self.M) ], dim=0) # [M. K], in some case(CUSTOM_KERNEL)
                attn_map = torch.cat([self.codebook[_][0].kernel for _ in range(self.M)],dim=0) # [M. K]
                self.register_buffer('attn_map', attn_map)
                self.register_buffer('choice_map', choice)

            elif self.qk_type == 'pairwise':

                q_ = self.q(x)
                q_f = q_.F
                N, _ = q_.F.shape

                choices = []
                for i_m, kg in enumerate(self.kgargs):  # iter over M

                    # codebook_weight = []
                    # for _ in range(self.M):
                        # codebook_weight.append(self.codebook[_][0].kernel)
                    # codebook_weight = torch.stack(codebook_weight, dim=-1) # [K, vec_dim, M]  
                    if 'dimension' in kg.keys():
                        del kg['dimension']
                    neis_d = q_.coordinate_manager.get_kernel_map(q_.coordinate_map_key,
                                                                    q_.coordinate_map_key,
                                                                    **kg
                                                                        )

                    choice = []
                    for k_ in range(len(neis_d.keys())):

                        if not k_ in neis_d.keys():
                            continue

                        neis_ = torch.gather(q_.F, dim=0, index=neis_d[k_][0].reshape(-1,1).expand(-1,self.vec_dim).long())
                        neis = torch.zeros(N,self.vec_dim, device=q_.F.device)  # DEBUG: not sure if needs decalre every time
                        neis.scatter_(dim=0, index=neis_d[k_][1].reshape(-1,1).expand(-1,self.vec_dim).long(), src=neis_)

                        sparse_mask_cur_k = (neis.abs().sum(-1) > 0).float()
                        # neis = neis - (q_.F*sparse_mask_cur_k.unsqueeze(-1).expand(-1, self.vec_dim))
                        # neis = self.map_qk(neis)  # apply a linear layer over the neighbor
                        neis = neis*(q_.F*sparse_mask_cur_k.unsqueeze(-1).expand(-1, self.vec_dim))
                        neis = neis*sparse_mask_cur_k.unsqueeze(-1).expand(-1, self.vec_dim)

                        out_cur_k = self.expand_vec_dim(neis)*self.codebook[i_m][0].kernel[k_].unsqueeze(0)
                        out_cur_k = out_cur_k.sum(1)  # [N]

                        choice.append(out_cur_k)

                    # choice = F.softmax(torch.stack(choice, dim=-1), dim=-1).sum(-1)
                    choice = torch.stack(choice, dim=-1)  # [N,K]
                    choice = F.softmax(choice/self.temp, dim=-1).sum(-1)
                    choices.append(choice) # [N]

                choices = torch.stack(choices, dim=-1)
                choices = F.softmax(choices/self.temp, dim=-1)   # [N,M]

                # if self.smooth_choice:
                    # # trying use avg_pool instead of conv
                    # choice = ME.SparseTensor(features=choice, coordinate_map_key=q_.coordinate_map_key, coordinate_manager=q_.coordinate_manager) # [N, dim]
                    # choice = self.smooth_conv(choice).F
                choice = choices.unsqueeze(1).expand(-1, self.vec_dim, -1) # [N, dim, M]
                self.register_buffer('choice_map', choices)
                # self.register_buffer('coord_map')

        if self.skip_choice:
            N, dim = v_.shape
            # choice = torch.randint(0,self.M, (N,1,self.M))
            # choice = torch.ones((N,1,self.M), device=v_.device).fill_(1/self.M)
            out = []
            for _ in range(self.M):
                self.codebook[_][0].kernel.requires_grad = True
                out_ = self.codebook[_](v_)
                out.append(out_.F)
            out = torch.stack(out, dim=-1)
            out = out.sum(-1)

        elif self.top_k_choice:
            # DEV: maybe add Channel Choice also to support a point with different vec_dims(needs repeating), maybe even nearly full(since if a point have many choices, it will appear in many choices)
            assert self.vec_dim == 1 # same point use the same choice 
            out = torch.zeros([N,dim,self.top_k_choice], device=x.device)
            choice_topk = torch.topk(choice, self.top_k_choice, dim=-1)[0] # shape [N,dim]
            choice_topk_idx = torch.topk(choice, self.top_k_choice, dim=-1)[1][:,0,:]  # shape [N]
            for _ in range(self.M):
                self.codebook[_][0].kernel.requires_grad = True
                # DEV: split points for different choice
                # however, if choice has the channle freedom
                # could not handle
                cur_out_ = self.codebook[_](v_) # the conv
                for top_ in range(self.top_k_choice):
                    choice_idx = torch.where(choice_topk_idx[:,top_] == _)[0]
                    # cur_v_ = v_.features_at_coordinates(v_.C[choice_idx,:].float())
                    if len(choice_idx) > 1:
                        # DEBUG: whether this will cause no grad back to v_?
                        # cur_v_ = ME.SparseTensor(
                                # features=v_.F[choice_idx,:],
                                # coordinates=v_.C[choice_idx,:],
                                # coordinate_map_key=v_.coordinate_map_key,
                                # coordinate_manager=v_.coordinate_manager
                                # )
                        out[:,:,top_].scatter_(
                                src=cur_out_.F[choice_idx,:]*choice_topk[choice_idx,:,top_],
                                # src=cur_out_.F
                                index=choice_idx.unsqueeze(-1).repeat(1,dim),
                                dim=0)
                    else:
                        pass
            out = out.sum(-1)
        else:
            # normal-case: apply the attn_weight aggregation with the channelwiseConvolution
            out = torch.zeros([N, self.planes*self.h], device=v_.device)
            for _ in range(self.M):
                self.codebook[_][0].kernel.requires_grad = True
                out_ = self.codebook[_](v_)
                out += out_.F*self.expand_vec_dim(choice[:,:,_])
                # out.append(out_.F)
            # out = torch.stack(out, dim=-1)
            # out = (out*self.expand_vec_dim(choice)).sum(-1)
            out = out.reshape([N, self.planes*self.h])

        out = ME.SparseTensor(features=out, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        out = self.out_bn_relu(out)
        out = out + res

        if self.with_label_embedding:
            self.get_label_embedding(out, aux)

        return out


class MultiHeadDiscreteAttnTRBlock(nn.Module):
    expansion=1
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
                   D=3,
                   ):
        super(MultiHeadDiscreteAttnTRBlock, self).__init__()

        self.h = 4
        self.blocks = nn.ModuleList([])
        for _ in range(self.h):
            self.blocks.append(
                DiscreteAttnTRBlock(
                    inplanes,
                    planes,
                    )
                )
        self.final_mapping = MinkoskiConvBNReLU(planes*self.h, planes, kernel_size=1)

    def forward(self, x, iter_=None, aux=None):
        outs = []
        for _ in range(self.h):
            outs.append(self.blocks[_](x).F)
        outs = torch.cat(outs, dim=-1)
        outs = ME.SparseTensor(features=outs, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        outs = self.final_mapping(outs)
        return outs

class ConvTRBlock(nn.Module):
    expansion=1
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
                   D=3,
                   ):
        super(ConvTRBlock, self).__init__()

        self.h = 1
        self.vec_dim = 4
        self.alphas = nn.Parameter(torch.rand([2]))
        self.tr = TRBlock(
                    inplanes,
                    planes,
                    )
        self.conv = BasicBlock(
                    inplanes,
                    planes,
                    downsample=downsample,
                )

    def forward(self, x, iter_=None, aux=None):
        try:
            outs = self.conv(x)*self.alphas[0] + self.tr(x)*self.alphas[1]
        except:
            import ipdb; ipdb.set_trace()
        return outs


# class DiscreteQKTRBlock(TRBlock):

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

        # super(DiscreteQKTRBlock, self).__init__(
                # inplanes,
                # planes,
                # )

        # self.type = 'discrete_qk'
        # '''
        # the discrete_qk version of the TR
        # Q and its neighbor has discrete actiavtion from the codebook
        # they substract and linear-mapped into the attention_weight
        # which is applied on the value
        # (possible hard to train)
        # ------------------
        # qk_type:
            # - conv
            # - substration
        # conv_v: use conv or linear for gen value
        # vec_dim: the attn_map feature dim
        # M - codebook size
        # temp - the softmax temperature
        # '''

        # # self.M = 12
        # # self.M = 4
        # self.expansion=1
        # self.M = 1
        # self.qk_type = 'dot'
        # # self.qk_type = 'conv'
        # self.conv_v = False
        # # self.vec_dim = 1
        # self.vec_dim = planes // 8
        # # self.vec_dim = 8
        # self.temp = 1
        # # self.top_k_choice = self.M
        # self.top_k_choice = False
        # self.neighbor_type = 'sparse_query'
        # self.k = 27

        # self.vq_loss = 0.
        # self.vq_lambda = 1.e-3
        # self.vq_loss_commit_beta= 0.5

        # # === some additonal tricks ===
        # self.skip_choice = False # only_used in debug mode, notice that this mode contains unused params, so could not work with ddp
        # self.smooth_choice = False

        # if self.inplanes != self.planes:
            # self.linear_top = MinkoskiConvBNReLU(inplanes, planes, kernel_size=1)
            # self.downsample = ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, dimension=3)

        # if self.qk_type == 'conv':
            # # since conv already contains the neighbor info, so no pos_enc
            # self.q = MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=3)
        # elif self.qk_type == 'pairwise':
            # self.q = MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=3)
            # # self.q = MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=3)
            # self.map_qk = nn.Linear(self.vec_dim, self.vec_dim)
        # elif self.qk_type == 'dot':
            # self.q = MinkoskiConvBNReLU(planes, self.vec_dim, kernel_size=3)
            # self.map_choice = nn.Sequential(
                    # nn.Linear(self.planes, self.planes),
                    # nn.ReLU(),
                # )
        # else:
            # raise NotImplementedError

        # if self.conv_v == True:
            # self.v = MinkoskiConvBNReLU(planes, planes*self.expansion, kernel_size=3)
        # else:
            # self.v = MinkoskiConvBNReLU(planes, planes*self.expansion, kernel_size=1)
        # self.codebook = nn.Parameter(
                # # torch.nn.init.xavier_uniform_(
                # torch.nn.init.kaiming_uniform_(
                    # torch.empty(self.planes, self.M), nonlinearity='relu'
                # ) / 10 # / 10  to make the data around 1e-2, which is important for training!
            # )
        # self.out_bn_relu = nn.Sequential(
                # ME.MinkowskiConvolution(planes*self.expansion, planes, kernel_size=1, dimension=3),
                # ME.MinkowskiBatchNorm(planes),
                # ME.MinkowskiReLU(),
                # )

        # if self.smooth_choice:
            # self.smooth_conv = ME.MinkowskiChannelwiseConvolution(self.M, kernel_size=3, dimension=3) # the fixed low-pass filter for local neighbors

    # def forward(self, x, iter_=None):
        # '''
        # TODO:
        # 1st do qk projection: [N, dim, k]
                # - conv: directly use conv neighbor aggregation(extra params), output: [N, vec_dim]
                # - substract: use linear mapping, then gather neighbor & substract. output: [N, vec_dim, k] -> [N, vec_dim] (requires extra transform, memory-hungry)
        # 2nd: q_ do dot product with M set of conv weights(apply conv): [N, dim, M] -> [N, dim, M], the apply softmax
            # - (q may be vec_dim instead of dim, broadcast to dims)
        # 3rd: use attn_map: [N, M] to aggregate M convs for each point
        # '''

        # if self.planes != self.inplanes:
            # res = self.downsample(x)
            # x = self.linear_top(x)
        # else:
            # res = x

        # v_ = self.v(x)

        # if self.qk_type == 'conv':
            # raise NotImplementedError

        # elif self.qk_type == 'pairwise':
            # raise NotImplementedError

        # elif self.qk_type == 'dot':
            # q_ = self.q(x)
            # q_f = q_.F # the continous q: [N, vec_dim]

            # N, _  = q_f.shape
            # dim, M = self.codebook.shape


            # index = torch.tensor(list(range(self.vec_dim))*(self.planes // self.vec_dim), device=q_f.device).long()
            # q_f = q_f.index_select(-1, index)

            # # get_vq_loss diffs as l2_reg
            # self.vq_loss = 0. # clear every iter
            # for m_ in range(self.M):
                # diff_a = (self.codebook[None,:,m_].detach() - q_f).abs().sum()
                # diff_b = (self.codebook[None,:,m_] - q_f.detach()).abs().sum()
                # self.vq_loss = self.vq_loss + diff_a + self.vq_loss_commit_beta*diff_b
            # self.vq_loss = self.vq_lambda*self.vq_loss/N

            # discrete_q = GetCodebookWeightStraightThroughQK.apply(self,q_f)  # [N, dim]  

            # if self.smooth_choice:
                # dummy_kernel = nn.Parameter(torch.ones_like(self.smooth_conv.kernel))
                # self.smooth_conv.kernel = dummy_kernel
                # self.smooth_conv.kernel.requires_grad = False
                # # choice = self.smooth_conv(choice)

            # neis_d = q_.coordinate_manager.get_kernel_map(q_.coordinate_map_key,
                                                            # q_.coordinate_map_key,
                                                            # kernel_size=3,
                                                            # stride=1,
                                                            # )
            # # assert q_.coordinate_manager == v_.coordinate_manager

            # out_qk = []
            # for k_ in range(self.k):

                # if not k_ in neis_d.keys():
                    # continue

                # neis_ = torch.gather(discrete_q, dim=0, index=neis_d[k_][0].reshape(-1,1).expand(-1,self.planes).long())
                # neis = torch.zeros(N,self.planes, device=q_.F.device)  # DEBUG: not sure if needs decalre every time
                # neis = torch.scatter(neis, dim=0, index=neis_d[k_][1].reshape(-1,1).expand(-1,self.planes).long(), src=neis_)
                # sparse_mask_cur_k = (neis.abs().sum(-1) > 0).float()
                # sparse_mask_cur_k = sparse_mask_cur_k.unsqueeze(-1).expand(-1, self.planes)
                # # DEBUG: currently only support attn_dim=1
                # out_cur_k = (neis*discrete_q*sparse_mask_cur_k).sum(-1)

                # out_qk.append(out_cur_k)  # [N, 1]

            # out_qk = torch.stack(out_qk, dim=-1)  # [N, k, 1]
            # out_qk = F.softmax(out_qk/self.temp, dim=-1) # debug_only

            # if self.skip_choice:
                # dummy_qk = torch.ones_like(out_qk)

            # out = torch.zeros_like(v_.F)
            # for k_ in range(self.k):

                # if not k_ in neis_d.keys():
                    # continue

                # neis_ = torch.gather(v_.F, dim=0, index=neis_d[k_][0].reshape(-1,1).expand(-1,self.planes).long())
                # neis = torch.zeros(N,self.planes, device=q_.F.device)  # DEBUG: not sure if needs decalre every time
                # # neis = torch.scatter(neis, dim=0, index=neis_d[k_][1].reshape(-1,1).expand(-1,self.planes).long(), src=neis_)
                # neis.scatter_(dim=0, index=neis_d[k_][1].reshape(-1,1).expand(-1,self.planes).long(), src=neis_)
                # sparse_mask_cur_k = (neis.abs().sum(-1) > 0).float()
                # sparse_mask_cur_k = sparse_mask_cur_k.unsqueeze(-1).expand(-1, self.planes)
                # # DEBUG: currently only support attn_dim=1
                # if self.skip_choice:
                    # out_cur_k = (neis*dummy_qk[:,k_].unsqueeze(-1))
                # else:
                    # out_cur_k = (neis*out_qk[:,k_].unsqueeze(-1))
                # out += out_cur_k

        # out = ME.SparseTensor(features=out, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        # out = self.out_bn_relu(out)
        # out = out + res

        # return out









