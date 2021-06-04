import itertools
import operator
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

import numpy as np
import numpy.ma as ma

from pointnet2_utils import furthest_point_sample as farthest_point_sample_cuda
from pointnet2_utils import gather_operation as index_points_cuda_transpose
from pointnet2_utils import grouping_operation as grouping_operation_cuda
from pointnet2_utils import ball_query as query_ball_point_cuda
from pointnet2_utils import QueryAndGroup
from pointnet2_utils import three_nn
from pointnet2_utils import three_interpolate

from knn_cuda import KNN
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points_cuda(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    points = points.transpose(1,2).contiguous() #[B, C, N]
    new_points = index_points_cuda_transpose(points, idx) #[B, C, S]

    return new_points.transpose(1,2).contiguous()


def stem_knn(xyz, points, k):
    knn = KNN(k=k, transpose_mode=True)
    xyz = xyz.permute([0,2,1])
    _, idx = knn(xyz.coniguous(), xyz) # xyz: [bs, npoints, coord] idx: [bs, npoint, k]
    idx = idx.int()

    # take in [B, 3, N]
    grouped_xyz = grouping_operation_cuda(xyz.transpose(1,2).contiguous(), idx) # [bs, xyz, n_point, k]
    grouped_points = grouping_operation_cuda(points.contiguous(), idx) #B, C, npoint, k)

    return grouped_xyz, grouped_points


def sample_and_group_cuda(npoint, k, xyz, points, cat_xyz_feature=True, fps_only=False):
    """
    Input:
        npoint:
        k:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, C, N]
    Return:
        new_xyz: sampled points position data, [B, 3, npoint]
        new_points: sampled points data, [B, C+C_xyz, npoint, k]
        grouped_xyz_norm: sampled relative points position data, [B, 3, npoint, k]
    """
    k = min(npoint, k)
    knn = KNN(k=k, transpose_mode=True)

    B, N, C_xyz = xyz.shape

    if npoint < N:
        # fps_idx = torch.arange(npoint).repeat(xyz.shape[0], 1).int().cuda() # DEBUG ONLY
        fps_idx = farthest_point_sample_cuda(xyz, npoint) # [B, npoint]
        torch.cuda.empty_cache()
        new_xyz = index_points_cuda(xyz, fps_idx) #[B, npoint, 3]
        new_points = index_points_cuda(points.transpose(1,2), fps_idx)
    else:
        new_xyz = xyz

    if fps_only:
        return new_xyz.transpose(1,2), new_points.transpose(1,2), fps_idx

    torch.cuda.empty_cache()
    _, idx = knn(xyz.contiguous(), new_xyz) # B, npoint, k
    idx = idx.int()

    torch.cuda.empty_cache()
    grouped_xyz = grouping_operation_cuda(xyz.transpose(1,2).contiguous(), idx).permute(0,2,3,1) # [B, npoint, k, C_xyz]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C_xyz) # [B, npoint, k, 3]
    grouped_xyz_norm = grouped_xyz_norm.permute(0,3,1,2).contiguous()# [B, 3, npoint, k]
    torch.cuda.empty_cache()

    grouped_points = grouping_operation_cuda(points.contiguous(), idx) #B, C, npoint, k

    if cat_xyz_feature:
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1) # [B, C+C_xyz, npoint, k]
    else:
        new_points = grouped_points # [B, C+C_xyz, npoint, k]

    return new_xyz.transpose(1,2), grouped_xyz_norm, new_points, idx

def voxel2points(x: ME.SparseTensor):
    '''
    pack the ME Sparse Tensor feature(batch-dim information within first col of coord)
    [N_voxel_all_batches, dims] -> [bs, max_n_voxel_per_batch, dim]

    idx are used to denote the mask
    '''

    x_c, mask, idx = separate_batch(x.C)
    B = x_c.shape[0]
    N = x_c.shape[1]
    dim = x.F.shape[1]
    idx_ = idx.reshape(-1,1).repeat(1,dim)
    x_f = torch.zeros(B*N, dim).cuda()
    x_f.scatter_(dim=0, index=idx_, src=x.F)
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

class TDLayer(nn.Module):
    def __init__(self, input_dim, out_dim, k=16, kernel_size=2):
        super().__init__()
        '''
        Transition Down Layer
        npoint: number of input points
        nsample: k in kNN, default 16
        in_dim: feature dimension of the input feature x (output of the PCTLayer)
        out_dim: feature dimension of the TDLayer
        '''
        self.k = k
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size

        '''a few additional cfg for TDLayer'''
        self.POINT_TR_LIKE = False
        self.FPS_ONLY = True
        self.cat_xyz_feature = True

        if self.POINT_TR_LIKE:
            if self.FPS_ONLY:
                # DEBUG: use 3x3 downsample
                # if self.cat_xyz_feature:
                    # self.projection = nn.Sequential(
                        # nn.Conv2d(input_dim+3, out_dim, 1),
                        # nn.BatchNorm2d(out_dim),
                        # nn.Conv2d(out_dim, out_dim, 1),
                        # nn.BatchNorm2d(out_dim),
                            # )
                # else:
                    # self.projection = nn.Sequential(
                        # nn.Conv2d(input_dim, out_dim, 1),
                        # nn.BatchNorm2d(out_dim),
                        # nn.ReLU(),
                        # # nn.Conv2d(out_dim, out_dim, 1),
                        # # nn.BatchNorm2d(out_dim),
                            # )
                if self.cat_xyz_feature:
                    self.conv = nn.Sequential(
                        ME.MinkowskiConvolution(input_dim+3, out_dim, kernel_size=1, bias=True, dimension=3),
                        ME.MinkowskiBatchNorm(out_dim),
                        ME.MinkowskiReLU(),
                        )
                else:
                    self.conv = nn.Sequential(
                        ME.MinkowskiConvolution(input_dim, out_dim, kernel_size=1, bias=True, dimension=3),
                        ME.MinkowskiBatchNorm(out_dim),
                        ME.MinkowskiReLU(),
                        )


            else:
                self.mlp_convs = nn.ModuleList()
                self.mlp_bns = nn.ModuleList()

                if self.cat_xyz_feature:
                    self.mlp_convs.append(nn.Conv2d(input_dim+3, input_dim, 1))
                else:
                    self.mlp_convs.append(nn.Conv2d(input_dim, input_dim, 1))
                self.mlp_convs.append(nn.Conv2d(input_dim, out_dim, 1))
                self.mlp_bns.append(nn.BatchNorm2d(input_dim))
                self.mlp_bns.append(nn.BatchNorm2d(out_dim))

                        # DEBUG ONLY
            # self.strided_conv = nn.Sequential(
                # ME.MinkowskiConvolution(input_dim,out_dim,kernel_size=1,stride=2,bias=True,dimension=3),
                # ME.MinkowskiBatchNorm(out_dim),
                # ME.MinkowskiReLU()
            # )

        else:
            self.conv = nn.Sequential(
                ME.MinkowskiConvolution(input_dim,out_dim,kernel_size=1,stride=2,bias=True,dimension=3),
                ME.MinkowskiBatchNorm(out_dim),
                ME.MinkowskiReLU()
            )

    def forward(self, x : ME.SparseTensor):
        """
        Input:
            xyz: input points position data, [B, 3, N]
            points: input points data, [B, C, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        if self.POINT_TR_LIKE:
            x_c, mask, idx = separate_batch(x.C)
            B = x_c.shape[0]
            N = x_c.shape[1]
            dim = x.F.shape[1]
            idx_ = idx.reshape(-1,1).repeat(1,dim)
            x_f = torch.zeros(B*N, dim).cuda()
            x_f.scatter_(dim=0, index=idx_, src=x.F)
            x_f = x_f.reshape([B,N,dim])

            k = 16
            ds_ratio = 4
            npoint = N//ds_ratio
            # x_c = x_c.transpose(1,2).float()
            x_f = x_f.transpose(1,2)

            if self.FPS_ONLY:
                # just using the FPS's result for subsample, without projection
                new_xyz, new_points, fps_idx = sample_and_group_cuda(npoint, k, x_c.float(), x_f, cat_xyz_feature=self.cat_xyz_feature, fps_only=True)
                if self.cat_xyz_feature:
                    additional_xyz = new_xyz / new_xyz.mean()
                    new_points_pooled = torch.cat([new_xyz/new_xyz.mean(),new_points], dim=1) # norm the xyz to some extent
                else:
                    new_points_pooled = new_points

                # DEBUG: debug here, skip the projection and project on the voxel
                # new_points_pooled = self.projection(new_points_pooled.unsqueeze(-1)).squeeze(-1) # [N,dim,nvoxel,1]

                B, new_dim, new_N = list(new_points_pooled.shape)

                # TODO: CK missing mask here：may create some points which is not there(as 0,0,0)
                # idx: [N-voxel] -> value in range(0, B*N)
                # fps_idx: [B,N//2] -> check whether in idx(transform): -> fps_mask: [<B*N/2] value: [0, B,N//2]

                batch_ids = torch.arange(B).unsqueeze(-1).repeat(1,new_N).reshape(-1,1).cuda()
                # fps_idx_full = batch_ids.view(-1)*N + fps_idx.view(-1)
                # # DEBUG: pytorch has no isin() function, parallel compare is memory-hungry
                # # and the iter version is slow
                # fps_in_mask = np.in1d(fps_idx_full.cpu().numpy(), idx.cpu().numpy().astype(int))
                # # the +1 and -1 are for the 0-idx
                # new_fps_idx_full = (((fps_idx_full.cpu().numpy()+1)*fps_in_mask).nonzero()[0]-1)
                # if len(new_fps_idx_full) < len(fps_idx_full):
                    # print('detected masked point!')
                    # import ipdb; ipdb.set_trace()

                # if no masked points are sampled, we could simply use a full-idx to gather new_feature
                new_idx = torch.arange(B*new_N).cuda()
                new_xyz = torch.gather(new_xyz.transpose(1,2).reshape(B*new_N, 3), dim=0, index=new_idx.reshape(-1,1).repeat(1,3))
                new_xyz = torch.cat([batch_ids, new_xyz], dim=1)
                new_points_pooled = torch.gather(new_points_pooled.transpose(1,2).reshape(B*new_N, new_dim), dim=0, index=new_idx.reshape(-1,1).repeat(1,new_dim))

            else:
                new_xyz, grouped_xyz_norm, new_points, new_indices  = sample_and_group_cuda(npoint, k, x_c.float(), x_f, cat_xyz_feature=self.cat_xyz_feature)

                # --- make the new idx, and ck if all new coord in old_coord ---
                # to_sum = (torch.arange(B).reshape(-1,1)*N).cuda() # the batch-dim
                # new_idx = torch.sort(new_indices[:,:,0],dim=-1)[0]
                # new_idx = new_idx + to_sum
                # new_idx = new_idx.view(-1)  # should be roughly half the size of the 'idx'
                # # there should not be a outlier point
                # ck_in = [not i in idx for i in new_idx] # didnt find a torch func to do that
                # assert sum(ck_in) == 0

                for i, conv in enumerate(self.mlp_convs):
                    bn = self.mlp_bns[i]
                    new_points =  F.relu(bn(conv(new_points)))

                new_points_pooled = torch.max(new_points, 3)[0]

                B, new_dim, new_N = new_points_pooled.shape
                new_idx = torch.arange(B*new_N).cuda()  # the keep all idxs
                # TODO: 
                new_points_pooled = new_points_pooled.transpose(1,2).reshape(B*new_N, new_dim)
                new_points_pooled = torch.gather(new_points_pooled, dim=0, index=new_idx.reshape(-1,1).repeat(1,new_dim))
                new_xyz = torch.gather(new_xyz.transpose(1,2).reshape(B*new_N, 3), dim=0, index=new_idx.reshape(-1,1).repeat(1,3))
                batch_ids = torch.arange(B).unsqueeze(-1).repeat(1,new_N).reshape(-1,1).cuda()
                new_xyz = torch.cat([batch_ids, new_xyz], dim=1)

            y = ME.SparseTensor(features=new_points_pooled,coordinates=new_xyz,coordinate_manager=x.coordinate_manager)

            if self.FPS_ONLY:
                y = self.conv(y)

            # y_test = self.strided_conv(x)
            # d = {}
            # d['origin'] = x.C
            # d['FPS'] = y.C
            # d['stride'] = y_test.C
            # torch.save(d, 'sample.pth')
            # print(x.C.shape, y.C.shape, y_test.C.shape)
            # import ipdb; ipdb.set_trace()

        else:
            y = self.conv(x)

        return y

class TULayer(nn.Module):
    def __init__(self, input_a_dim, input_b_dim, out_dim,k=3):
        super().__init__()
        '''
        Transition Up Layer
        npoint: number of input points
        nsample: k in kNN, default 3
        input_a_dim: feature dimension of the input a(needs upsampling)
        input_b_dim: feature dimension of the input b (directly concat)
        out_dim: feature dimension of the TDLayer(fixed as the input_a_dim // 2) + input_b_dim

        '''
        self.k = k
        self.input_a_dim = input_a_dim
        self.input_b_dim = input_b_dim
        self.intermediate_dim = (input_a_dim // 2) + input_b_dim
        self.out_dim = out_dim

        self.POINT_TR_LIKE = False
        self.SUM_FEATURE = True # only used when POINTTR_LIKE is False

        # -------- Point TR like -----------
        if self.POINT_TR_LIKE:
            self.linear_a = nn.Linear(input_a_dim, out_dim)
            self.linear_b = nn.Linear(input_b_dim, out_dim)
        else:
            if self.SUM_FEATURE:
                self.conv_a = nn.Sequential(
                        ME.MinkowskiConvolutionTranspose(in_channels=input_a_dim, out_channels=out_dim,kernel_size=2,stride=2,dimension=3),
                        ME.MinkowskiBatchNorm(out_dim),
                        ME.MinkowskiReLU(),
                        )
                self.conv_b = nn.Sequential(
                        ME.MinkowskiConvolutionTranspose(in_channels=input_b_dim, out_channels=out_dim,kernel_size=1,stride=1,dimension=3),
                        ME.MinkowskiBatchNorm(out_dim),
                        ME.MinkowskiReLU(),
                        )
            else:
                self.conv = ME.MinkowskiConvolutionTranspose(
                    in_channels=input_a_dim,
                    out_channels=input_a_dim // 2,
                    kernel_size=2,
                    stride=2,
                    dimension=3
                )
                self.bn = ME.MinkowskiBatchNorm(
                    self.input_a_dim // 2
                )
                self.relu = ME.MinkowskiReLU()

                # -----------------------------------------

                self.out_conv = ME.MinkowskiConvolution(
                    in_channels=input_a_dim//2 + input_b_dim,
                    out_channels=out_dim,
                    kernel_size=3,
                    stride=1,
                    dimension=3
                )
                self.out_bn = ME.MinkowskiBatchNorm(
                    self.out_dim
                )
                self.out_relu = ME.MinkowskiReLU()

    def forward(self, x_a : ME.SparseTensor, x_b: ME.SparseTensor):
        """
        Input:
            M < N
            xyz_1: input points position data, [B, 3, M]
            xyz_2: input points position data, [B, 3, N]
            points_1: input points data, [B, C, M]
            points_2: input points data, [B, C, N]

            interpolate xyz_2's coordinates feature with knn neighbor's features weighted by inverse distance

            TODO: For POINT_TR_LIKE, add support for no x_b is fed, simply upsample the x_a

        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        if self.POINT_TR_LIKE:

            dim = x_b.F.shape[1]
            assert dim == self.out_dim

            x_ac, mask_a, idx_a = separate_batch(x_a.C)
            B = x_ac.shape[0]
            N_a = x_ac.shape[1]
            x_af = torch.zeros(B*N_a, dim).cuda()
            idx_a = idx_a.reshape(-1,1).repeat(1,dim)
            x_af.scatter_(dim=0, index=idx_a, src=self.linear_a(x_a.F))
            x_af = x_af.reshape([B, N_a, dim])

            x_bc, mask_b, idx_b = separate_batch(x_b.C)
            B = x_bc.shape[0]
            N_b = x_bc.shape[1]
            x_bf = torch.zeros(B*N_b, dim).cuda()
            idx_b = idx_b.reshape(-1,1).repeat(1,dim)
            x_bf.scatter_(dim=0, index=idx_b, src=self.linear_b(x_b.F))
            x_bf = x_bf.reshape([B, N_b, dim])

            dists, idx = three_nn(x_bc.float(), x_ac.float())

            # dists_ = square_distance(x_bc, x_ac)
            # dists_, idx_ = dists_.sort(dim=-1)
            # dists, idx = dists_[:,:,:self.k], idx_[:,:,:self.k]

            # del dists_, idx_
            # torch.cuda.empty_cache()

            mask = (dists.sum(dim=-1)>0).unsqueeze(-1).repeat(1,1,3)

            dist_recip = 1.0 / (dists + 1e-1)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            weight = weight*mask  # mask the zeros part

            interpolated_points = three_interpolate(x_af.transpose(1,2).contiguous(), idx, weight).transpose(1,2) # [B, N_b, dim]
            # interpolated_points = torch.sum(grouping_operation_cuda(x_af.transpose(1,2).contiguous(), idx.int())*weight.view(B, 1, N_b, 3) ,dim=-1)  # [B, dim, N_b]
            # interpolated_points = interpolated_points.transpose(1,2) # [B, N_b, dim]
            out = interpolated_points + x_bf
            # DEBUG: only assign to the b-branch is ok?
            out = torch.gather(out.reshape(B*N_b,dim), dim=0, index=idx_b) # should be the same size with x_a.F
            x = ME.SparseTensor(features = out, coordinate_map_key=x_b.coordinate_map_key, coordinate_manager=x_b.coordinate_manager)

        else:
            if self.SUM_FEATURE:
                x_a = self.conv_a(x_a)
                x_b = self.conv_b(x_b)
                x = x_a + x_b
            else:
                x_a = self.conv(x_a)
                x_a = self.bn(x_a)
                x_a = self.relu(x_a)
                x = me.cat(x_a, x_b)
                x = self.out_conv(x)
                x = self.out_bn(x)
                x = self.out_relu(x)

        return x

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

class TransposeLayerNorm(nn.Module):

    def __init__(self, dim):
        super(TransposeLayerNorm, self).__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 3:
            # [bs, in_dim, npoints]
            pass
        elif len(x.shape) == 4:
            # [bs, in_dim, npoints, k]
            pass
        else:
            raise NotImplementedError

        return self.norm(x.transpose(1,-1)).transpose(1,-1)

class StackedPTBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, is_firstlayer=False, n_sample=16, r=10, skip_knn=False, kernel_size=1):
        super().__init__()

        self.block1 = PTBlock(in_dim, hidden_dim, is_firstlayer, n_sample, r, skip_knn, kernel_size)
        self.block2 = PTBlock(in_dim, hidden_dim, is_firstlayer, n_sample, r, skip_knn, kernel_size)

    def forward(self, x : ME.SparseTensor):
        x = self.block1(x)
        x = self.block2(x)
        return x

class PTBlock(nn.Module):
    # TODO: set proper r, too small will cause less points
    def __init__(self, in_dim, hidden_dim, is_firstlayer=False, n_sample=16, r=10, skip_knn=False, kernel_size=1):
        super().__init__()
        '''
        Point Transformer Layer

        in_dim: feature dimension of the input feature x
        out_dim: feature dimension of the Point Transformer Layer(currently same with hidden-dim)
        '''

        self.r = r # neighborhood cube radius
        self.skip_knn = skip_knn
        self.kernel_size = kernel_size


        self.kernel_size = 1 # DEBUG ONLY

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = self.hidden_dim
        self.vector_dim = self.out_dim // 1

        self.n_sample = n_sample

        self.USE_KNN = True

        # whether to use the vector att or the original attention
        self.use_vector_attn = True
        if not self.use_vector_attn:
            self.nhead = 4

        self.linear_top = nn.Sequential(
            ME.MinkowskiConvolution(in_dim, self.hidden_dim, kernel_size=self.kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(self.hidden_dim),
        )
        self.linear_down = nn.Sequential(
            ME.MinkowskiConvolution(self.out_dim, self.in_dim, kernel_size=self.kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(self.in_dim),
        )
        # feature transformations
        self.phi = nn.Sequential(
            ME.MinkowskiConvolution(self.hidden_dim, self.out_dim, kernel_size=self.kernel_size, dimension=3)
        )
        self.psi = nn.Sequential(
            ME.MinkowskiConvolution(self.hidden_dim, self.out_dim, kernel_size=self.kernel_size, dimension=3)
        )
        self.SKIP_ATTN=False
        if self.SKIP_ATTN:
            KERNEL_SIZE = 1
            self.alpha = nn.Sequential(
                    nn.Conv1d(self.in_dim, self.in_dim, KERNEL_SIZE),
                    nn.BatchNorm1d(self.in_dim),
                    nn.ReLU(),
                    nn.Conv1d(self.in_dim, self.hidden_dim, KERNEL_SIZE),
                    nn.BatchNorm1d(self.hidden_dim),
                    nn.ReLU(),
                    )

            # self.alpha = nn.Sequential(
                # ME.MinkowskiConvolution(self.hidden_dim, self.out_dim, kernel_size=self.kernel_size, dimension=3),
                # ME.MinkowskiBatchNorm(self.out_dim),
                # ME.MinkowskiReLU(),
                # ME.MinkowskiConvolution(self.out_dim, self.out_dim, kernel_size=self.kernel_size, dimension=3),
                # ME.MinkowskiBatchNorm(self.out_dim),
                # ME.MinkowskiReLU(),
                # )
        else:
            self.alpha = nn.Sequential(
                ME.MinkowskiConvolution(self.hidden_dim, self.out_dim, kernel_size=self.kernel_size, dimension=3)
            )

        self.gamma = nn.Sequential(
            nn.Conv1d(self.out_dim, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.vector_dim, 1),
            nn.BatchNorm1d(self.vector_dim),
        )

        self.delta = nn.Sequential(
                    nn.Conv2d(3, self.hidden_dim, 1),
                    nn.BatchNorm2d(self.hidden_dim),
                    nn.ReLU(),
                    nn.Conv2d(self.hidden_dim, self.out_dim, 1),
                    nn.BatchNorm2d(self.out_dim),
                )

        # self.delta = nn.Sequential(
            # nn.Conv2d(3, self.hidden_dim, 3, padding=1), # debug： whether using 3x3 or 1x1 linear embedding
            # nn.BatchNorm2d(self.hidden_dim),
            # nn.ReLU(),
            # nn.Conv2d(self.hidden_dim, self.out_dim, 3, padding=1),
            # nn.BatchNorm2d(self.out_dim),
        # )

    def forward(self, x : ME.SparseTensor, aux=None):
        '''
        input_p:  B, 3, npoint
        input_x: B, in_dim, npoint
        '''
        PT_begin = time.perf_counter()
        self.B = (x.C[:,0]).max().item() + 1 # batch size
        npoint, in_dim = tuple(x.F.size())
        self.k = min(self.n_sample, npoint)
        if not self.use_vector_attn:
            h = self.nhead

        res = x

        if self.skip_knn:
            # --- for debugging only ---
            x = self.linear_top(x)
            y = self.linear_down(x)
            return y+res

            # self.cube_query = cube_query(r=self.r, k=self.k)
            # neighbor, mask, idx_ = self.cube_query.get_neighbor(x, x)
            # x = self.linear_top(x)
            # new_x = get_neighbor_feature(neighbor, x)

            # y = get_neighbor_feature(neighbor, self.tmp_linear(x))
            # y = y.mean(dim=1)
            # y = ME.SparseTensor(features = y, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
            # y = self.linear_down(x)
            # return y+res

        # ------------------------------------------------------------------------------

        else:
            # self.USE_KNN = False
            self.cube_query = cube_query(r=self.r, k=self.k, knn=self.USE_KNN)

            # neighbor: [B*npoint, k, bxyz]
            # mask: [B*npoint, k]
            # idx: [B_nq], used for scatter/gather
            neighbor, mask, idx_ = self.cube_query.get_neighbor(x, x)

            # DEBUG ONLY
            # self.query_knn = cube_query(r=1000, k=self.k, knn=True)
            # neighbor_knn, mask, idx_ = self.query_knn.get_neighbor(x, x)
            # d = {}
            # batch_id = 0
            # point_id = 1000
            # batch_ends = torch.where(x.C[:,0] == batch_id)[0][-1]
            # d['voxels'] = x.C[:batch_ends,1:]
            # d['center'] = x.C[point_id,1:]
            # d['nei-ball'] = neighbor[point_id,:,1:]
            # d['nei-knn'] = neighbor_knn[point_id,:,1:]
            # torch.save(d, 'voxel.pth')

            self.register_buffer('neighbor_map', neighbor)
            self.register_buffer('input_map', x.C)

            # check for duplicate neighbor(not enough voxels within radius that fits k)
            # CHECK_FOR_DUP_NEIGHBOR=True
            # if CHECK_FOR_DUP_NEIGHBOR:
                # dist_map = (neighbor - neighbor[:,0,:].unsqueeze(1))[:,1:,:].abs()
                # num_different = (dist_map.sum(-1)>0).sum(-1) # how many out of ks are the same, of shape [nvoxel]
                # outlier_point = (num_different < int(self.k*1/2)-1).sum()
                # if not (outlier_point < max(npoint//10, 10)):  # sometimes npoint//100 could be 3
                    # pass
                    # logging.info('Detected Abnormal neighbors, num outlier {}, all points {}'.format(outlier_point, x.shape[0]))

            x = self.linear_top(x) # [B, in_dim, npoint], such as [16, 32, 4096]

            '''
            illustration on dimension notations:
            - B: batch size
            - nvoxel: number of all voxels of the whole batch
            - k: k neighbors
            - feat_dim: feature dimension, or channel as others call it
            - nvoxel_batch: the maximum voxel number of a single SparseTensor in the current batch
            '''

            '''Gene the pos_encoding'''
            try:
                relative_xyz = neighbor - x.C[:,None,:].repeat(1,self.k,1) # (nvoxel, k, bxyz), we later pad it to [B, xyz, nvoxel_batch, k]
            except RuntimeError:
                import ipdb; ipdb.set_trace()

            '''
            mask the neighbor when not in the same instance-class
            '''
            if aux is not None:
                neighbor_mask = aux.features_at_coordinates(neighbor.reshape(-1,4).float()).reshape(-1,self.k) # [N, k]
                neighbor_mask = (neighbor_mask - neighbor_mask[:,0].unsqueeze(-1) != 0).int()
                # logging.info('Cur Mask Ratio {}'.format(neighbor_mask.sum()/neighbor_mask.nelement()))

                neighbor_mask = torch.ones_like(neighbor_mask) - neighbor_mask
            else:
                neighbor_mask = None

            WITH_POSE_ENCODING = True
            if WITH_POSE_ENCODING:
                relative_xyz[:,0,0] = x.C[:,0] # get back the correct batch index, because we messed batch index in the subtraction above
                relative_xyz = pad_zero(relative_xyz, mask) # [B, xyz, nvoxel_batch, k]
                pose_tensor = self.delta(relative_xyz.float()) # (B, feat_dim, nvoxel_batch, k)
                pose_tensor = make_position_tensor(pose_tensor, mask, idx_, x.C.shape[0]) # (nvoxel, k, feat_dim)S

            if self.SKIP_ATTN:
                grouped_x = get_neighbor_feature(neighbor, x) # (nvoxel, k, feat_dim)
                if WITH_POSE_ENCODING:
                    alpha = self.alpha((grouped_x + pose_tensor).transpose(1,2))
                else:
                    alpha = self.alpha((grouped_x).transpose(1,2))
                y = alpha.max(dim=-1)[0]
                y = ME.SparseTensor(features = y, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)

                y = self.linear_down(y)
                return y+res

            phi = self.phi(x).F # (nvoxel, feat_dim)
            phi = phi[:,None,:].repeat(1,self.k,1) # (nvoxel, k, feat_dim)
            psi = get_neighbor_feature(neighbor, self.psi(x)) # (nvoxel, k, feat_dim)
            alpha = get_neighbor_feature(neighbor, self.alpha(x)) # (nvoxel, k, feat_dim)

            '''The Self-Attn Part'''
            if self.use_vector_attn:
                '''
                the attn_map: [vector_dim];
                the alpha:    [out_dim]
                attn_map = F.softmax(self.gamma(phi - psi + pos_encoding), dim=-1) # [B, in_dim, npoint, k], such as [16, 32, 4096, 16]
                y = attn_map.repeat(1, self.out_dim // self.vector_dim,1,1)*(alpha + pos_encoding) # multiplies attention weight
                self.out_dim and self.vector_dim are all 32 here, so y is still [16, 32, 4096, 16]
                y = y.sum(dim=-1) # feature aggregation, y becomes [B, out_dim, npoint]
                '''
                if WITH_POSE_ENCODING:
                    attn_map = F.softmax(self.gamma((phi - psi + pose_tensor).transpose(1,2)), dim=-1)
                else:
                    attn_map = F.softmax(self.gamma((phi - psi).transpose(1,2)), dim=-1)
                if WITH_POSE_ENCODING:
                    self_feat = (alpha + pose_tensor).permute(0,2,1) # (nvoxel, k, feat_dim) -> (nvoxel, feat_dim, k)
                else:
                    self_feat = (alpha).permute(0,2,1) # (nvoxel, k, feat_dim) -> (nvoxel, feat_dim, k)

                # use aux info and mask the  attn_map
                if neighbor_mask is not None:
                    attn_map = attn_map*(neighbor_mask.unsqueeze(1))

                y = attn_map.repeat(1, self.out_dim // self.vector_dim, 1, 1) * self_feat # (nvoxel, feat_dim, k)
                y = y.sum(dim=-1).view(x.C.shape[0], -1) # feature aggregation, y becomes (nvoxel, feat_dim)
                y = ME.SparseTensor(features = y, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
            else:
                phi = phi.permute([2,1,0]) # [out_dim, k, npoint]
                psi = psi.permute([2,0,1]) # [out_dim. npoint, k]
                attn_map = F.softmax(torch.matmul(phi,psi), dim=0) # [out_dim, k, k]
                alpha = (alpha+pose_tensor).permute([2,0,1])  # [out_dim, npoint, k]
                y = torch.matmul(alpha, attn_map)  # [out_dim, npoint, k]
                y = y.sum(-1).transpose(0,1)  # [out_dim. npoint]
                y = ME.SparseTensor(features = y, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)

            y = self.linear_down(y)

            self.register_buffer('attn_map', attn_map.detach().cpu().data) # pack it with nn parameter to save in state-dict

            return y+res

def make_position_tensor(pose_encoding : torch.Tensor, mask : torch.Tensor, idx_: torch.Tensor, nvoxel : int):
    """
    Mask positional encoding into k ME.SparseTensors

    Input:
        pose_encoding: (B, feat_dim, nvoxel_batch, k)
        batch_tensor:  (B, N)
    """

    assert idx_.shape[0] == nvoxel # the idx and the nvoxel should be the same

    B, feat_dim, nvoxel_batch, k = pose_encoding.shape
    pose_encoding = pose_encoding.permute(0, 2, 3, 1) # (B, feat_dim, nvoxel_batch, k) -> (B, nvoxel_batch, k, feat_dim)

    '''use idx to scatter the result'''
    masked_encoding = torch.gather(
        pose_encoding.reshape(-1, k, feat_dim),
        0,
        idx_.reshape(-1,1,1).repeat(1, k, feat_dim)
    ).reshape(nvoxel, k, feat_dim)
    return masked_encoding # (nvoxel, k, feat_dim)

def get_neighbor_feature(neighbor: torch.Tensor, x: ME.SparseTensor):
    """
        fetch neighbor voxel's feature tensor.
        Input:
            neighbor: torch.Tensor [B*npoint, k, xyz]
            x:        ME.SparseTensor
    """
    B_npoint, k, _  = tuple(neighbor.size())
    neighbor = neighbor.view(-1, 4).float() # [B*npoint*k, bxyz]
    features = x.features_at_coordinates(neighbor)
    _, dim = features.shape
    features = features.view(-1, k, dim)
    return features

def pad_zero(tensor : torch.Tensor, mask: torch.Tensor):
    '''
    input is [B*npoint, k, bxyz], we want [B, xyz, npoint, k]
    need to pad zero because each batch may have different voxel number
    B = int(max(tensor[:,0,0]).item() + 1)
    k = tuple(tensor.shape)[1]
    '''
    B, N = mask.shape
    _, k, bxyz = tensor.shape
    result = torch.zeros([B, N, k, 4], dtype=torch.int, device=tensor.device)
    pointer = 0
    for b_idx in range(B):
        nvoxel = mask.sum(-1)[b_idx]
        result[b_idx, :nvoxel, :, :] = tensor[pointer:pointer+nvoxel, :, :]
        pointer += nvoxel
    result = result[:,:,:,1:] # (B, N, k, 3)
    result = result.permute(0, 3, 1, 2) # (B, N, k, 3) -> (B, 3, N, k)
    return result

def manhattan_dist(dxyz: tuple):
    dxyz = [abs(v) for v in dxyz]
    return sum(dxyz[1:])

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

# cube query for sparse tensors
class cube_query(object):
    """
    Cube query for ME.SparseTensor
        ref  : ME.SparseTensor, coord dim = [B * nr, 3]
            reference sparse tensor
        query: ME.SparseTensor, coord dim = [B * nq, 3]
            query sparse tensor, whose neighbors we are look for
    return:
        result: torch.Tensor [B * nq, k, 4], 4 is B,x,y,z
        mask:   torch.Tensor [B * nq, k], zero means less than k neighbors

    __init__():
        input:
            r: cube query radius
            k: k neighbors
    """
    def __init__(self, r, k, knn=False):
        self.r = r
        self.k = k
        if knn:
            self.use_knn = True
            self.knn = KNN(k=k, transpose_mode=True)
        else:
            self.use_knn = False

    def get_neighbor(self, ref : ME.SparseTensor, query : ME.SparseTensor):
        B_nq, _ = query.C.shape

        coord = query.C # (N, 4)
        batch_info = coord[:,0]
        coord, mask, idx_ = separate_batch(coord) # (b, n, 3)
        b, n, _ = coord.shape

        if self.use_knn:
            _, idx = self.knn(coord.contiguous(), coord)
            grouped_coord = grouping_operation_cuda(coord.float().transpose(1,2).contiguous(), idx.int())
            result_padded = grouped_coord.permute([0,2,3,1])
        else:
            query_and_group_cuda = QueryAndGroup(radius=self.r, nsample=self.k, use_xyz=False)
            coord = coord.float()

            # CK the proper radius size
            # radius = 6000 / n
            # k = 8

            # query_idx = query_ball_point_cuda(self.r, self.k, coord, coord) # [bs, npoint, k]
            # self.knn = KNN(k=self.k, transpose_mode=True)
            # _, knn_idx = self.knn(coord.contiguous(), coord)
            # d = {}
            # d['data'] = coord
            # d['query'] = query_idx[0,100]
            # d['knn'] = knn_idx[0,100]
            # torch.save(d, './coord.pth')
            # import ipdb; ipdb.set_trace()

            idxs = query_and_group_cuda(
                xyz=coord,
                new_xyz=coord,
                features=coord.transpose(1,2).contiguous(),
            ) # idx: [bs, xyz, npoint, nsample]
            idxs = idxs.permute([0,2,3,1]) # idx: [bs, npoint, nsample, xyz]
            result_padded = idxs

        # unpad result (b, n, k, 3) -> (B_nq, k, 4) by applying mask
        result = torch.zeros([B_nq, self.k, 4], dtype=torch.int32, device=query.device)
        result[:,:,1:] = torch.gather(
            result_padded.reshape(-1, self.k, 3),
            0,
            idx_.reshape(-1, 1, 1).repeat(1, self.k, 3)
        )
        result[:,:,0] = batch_info.unsqueeze(-1).repeat(1, self.k)

        return result, mask, idx_

if __name__ == "__main__":
    import torch
    import MinkowskiEngine as ME
    feature = torch.tensor([[0.2, 0.3], [0.4, 0.5]])
    coord =torch.tensor([[0.6, 0.8, 0.3], [0.4, 0.3, 0.5]])
    x = ME.SparseTensor(
        features = feature,
        coordinates = ME.utils.batched_coordinates([coord / 0.1])
    )
    cq = cube_query(1, 2)
    result, mask = cq.get_neighbor(x, x)
