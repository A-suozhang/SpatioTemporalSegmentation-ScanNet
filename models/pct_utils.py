import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_utils import furthest_point_sample as farthest_point_sample_cuda
from pointnet2_utils import gather_operation as index_points_cuda_transpose
from pointnet2_utils import grouping_operation as grouping_operation_cuda
from pointnet2_utils import ball_query as query_ball_point_cuda
from pointnet2_utils import QueryAndGroup

import MinkowskiEngine as ME

from knn_cuda import KNN

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


# def stem_knn(xyz, points, k):
    # knn = KNN(k=k, transpose_mode=True)
    # xyz = xyz.permute([0,2,1])
    # _, idx = knn(xyz.contiguous(), xyz) # xyz: [bs, npoints, coord] idx: [bs, npoint, k]
    # idx = idx.int()
    
    # # take in [B, 3, N]
    # grouped_xyz = grouping_operation_cuda(xyz.transpose(1,2).contiguous(), idx) # [bs, xyz, n_point, k]
    # grouped_points = grouping_operation_cuda(points.contiguous(), idx) #B, C, npoint, k)

    # return grouped_xyz, grouped_points


def sample_and_group_cuda(npoint, k, xyz, points, cat_xyz_feature=True):
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
        fps_idx = farthest_point_sample_cuda(xyz, npoint) # [B, npoint]
        torch.cuda.empty_cache()
        new_xyz = index_points_cuda(xyz, fps_idx) #[B, npoint, 3]
    else:
        new_xyz = xyz

    torch.cuda.empty_cache()
    _, idx = knn(xyz.contiguous(), new_xyz) # B, npoint, k
    idx = idx.int()

    torch.cuda.empty_cache()
    grouped_xyz = grouping_operation_cuda(xyz.transpose(1,2).contiguous(), idx).permute(0,2,3,1) # [B, npoint, k, C_xyz]
    torch.cuda.empty_cache()
    try:
        # DEBUG: when using the mixed-trans, some last voxels may have less points
        grouped_xyz_norm = grouped_xyz - new_xyz.view(-1, min(npoint,N), 1, C_xyz) # [B, npoint, k, 3]
    except:
        import ipdb; ipdb.set_trace()
    grouped_xyz_norm = grouped_xyz_norm.permute(0,3,1,2).contiguous()# [B, 3, npoint, k]
    torch.cuda.empty_cache()

    grouped_points = grouping_operation_cuda(points.contiguous(), idx) #B, C, npoint, k

    if cat_xyz_feature:
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=1) # [B, C+C_xyz, npoint, k]
    else:
        new_points = grouped_points # [B, C+C_xyz, npoint, k]


    return new_xyz.transpose(1,2), grouped_xyz_norm, new_points

class TDLayer(nn.Module):
    def __init__(self, npoint, input_dim, out_dim, k=16):
        super().__init__()
        '''
        Transition Down Layer
        npoint: number of input points
        nsample: k in kNN, default 16
        in_dim: feature dimension of the input feature x (output of the PCTLayer)
        out_dim: feature dimension of the TDLayer

        '''
        self.npoint = npoint
        self.k = k
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        # by default the xyz information is concated to feature
        self.cat_xyz_feature=False
        if self.cat_xyz_feature:
            self.mlp_convs.append(nn.Conv2d(input_dim+3, input_dim, 1))
        else:
            self.mlp_convs.append(nn.Conv2d(input_dim, input_dim, 1))
        self.mlp_convs.append(nn.Conv2d(input_dim, out_dim, 1))
        self.mlp_bns.append(nn.BatchNorm2d(input_dim))
        self.mlp_bns.append(nn.BatchNorm2d(out_dim))

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, 3, N]
            points: input points data, [B, C, N]
        Return:
            gew_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B, input_dim, npoint = list(xyz.size())
        xyz = xyz.permute(0, 2, 1)

        FIXED_NUM_POINTS=False
        if FIXED_NUM_POINTS:
            npoint = self.npoint
        else:
            ds_ratio=2
            npoint = npoint // ds_ratio

        new_xyz, grouped_xyz_norm, new_points = sample_and_group_cuda(npoint, self.k, xyz, points, cat_xyz_feature=self.cat_xyz_feature)
        # new_xyz: sampled points position data, [B, 3, npoint]
        # new_points: sampled points data, [B, C+C_xyz, npoint,k]
        # grouped_xyz_norm: [B, 3, npoint,k]

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points_pooled = torch.max(new_points, 3)[0] # local max pooling
        # return new_xyz, new_points_pooled, grouped_xyz_norm, new_points
        return new_xyz, new_points_pooled

class TULayer(nn.Module):
    def __init__(self, npoint, input_dim, out_dim, k=3):
        super().__init__()
        '''
        Transition Up Layer
        npoint: number of input points
        nsample: k in kNN, default 3
        in_dim: feature dimension of the input feature x (output of the PCTLayer)
        out_dim: feature dimension of the TDLayer

        '''
        self.npoint = npoint
        self.k = k
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.linear_1 = nn.Conv1d(input_dim, out_dim, 1)
        self.linear_2 = nn.Conv1d(out_dim, out_dim, 1)

        self.CONCAT_FEATS = False
        # if use pointnet++ like concat_feats TU
        # linear before the interpolation, 
        # and use concat instead of sum for aggregation
        self.projection = nn.Sequential(
                nn.Conv1d(input_dim+out_dim, out_dim,1),
                nn.Conv1d(out_dim, out_dim,1),
                nn.Conv1d(out_dim, out_dim,1),
                )


    def forward(self, xyz_1, xyz_2, points_1, points_2):
        """
        Input:
            M < N
            xyz_1: input points position data, [B, 3, M]
            xyz_2: input points position data, [B, 3, N]
            points_1: input points data, [B, C, M]
            points_2: input points data, [B, C, N]

            interpolate xyz_2's coordinates feature with knn neighbor's features weighted by inverse distance

        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        B, input_dim, M = list(points_1.size())
        B, output_dim, N = list(points_2.size())

        if self.CONCAT_FEATS:
            pass
        else:
            points_1 = self.linear_1(points_1)
            points_2 = self.linear_2(points_2)

        dists = square_distance(xyz_2.transpose(1,2), xyz_1.transpose(1,2)) # [B, N, M]
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:,:,:self.k], idx[:,:,:self.k]

        dist_recip = 1.0 / (dists + 1e-1)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum( \
                        grouping_operation_cuda(points_1, idx.int())*weight.view(B, 1, N, 3)
                                                ,dim=-1)
        if self.CONCAT_FEATS:
            new_points = torch.cat([interpolated_points, points_2], dim=1)
            new_points = self.projection(new_points)
            return xyz_2 , new_points

        return xyz_2 , (interpolated_points + points_2)

# def index_points(points, idx):
    # """
    # Input:
        # points: input points data, [B, N, C]
        # idx: sample index data, [B, S, [K]]
    # Return:
        # new_points:, indexed points data, [B, S, [K], C]
    # """
    # raw_size = idx.size()
    # idx = idx.reshape(raw_size[0], -1)
    # res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    # return res.reshape(*raw_size, -1)

# class TransposeLayerNorm(nn.Module):

    # def __init__(self, dim):
        # super(TransposeLayerNorm, self).__init__()
        # self.dim = dim
        # self.norm = nn.LayerNorm(dim)

    # def forward(self, x):
        # if len(x.shape) == 3:
            # # [bs, in_dim, npoints]
            # pass
        # elif len(x.shape) == 4:
            # # [bs, in_dim, npoints, k]
            # pass
        # else:
            # raise NotImplementedError

        # return self.norm(x.transpose(1,-1)).transpose(1,-1)

class TRBlock(nn.Module):
    def __init__(self, in_dim, expansion=1, n_sample=16, fps_rate=None, radius=None):
        super().__init__()
        self.block0 = PTBlock(in_dim, expansion=expansion, n_sample=n_sample, fps_rate=fps_rate, radius=radius)
        # the normal block, no expansion and downsample
        self.block1 = PTBlock(in_dim*expansion, expansion=1, n_sample=n_sample, fps_rate=None, radius=radius)

    def forward(self, input_p, input_x):
        input_p_reduced, input_x_reduced = self.block0(input_p, input_x)
        output_p, output_x = self.block1(input_p_reduced, input_x_reduced)
        return output_p, output_x

class PTBlock(nn.Module):
    def __init__(self, in_dim, expansion=1, n_sample=16, fps_rate=None, radius=None):
        super().__init__()
        '''
        --- Point Transformer Layer ---

        in_dim: feature dimension of the input feature x
        hidden_dim: feature after the linear-top, normally = in_dim, could be different
        out_dim: feature dimension of the Point Transformer Layer(qkv dim)
        vector_dim: the dim of the vector attn, should be out_dim/N, set here same as out_dim

        linear_top: in_dim -> hidden
        qkv: hidden -> out(vector)
        gamma: out -> vector (qkv as its input, attn_map as its output)
        pos_encoding: hidden -> out
        linear_down: out_dim -> in_dim
        '''
        self.in_dim = in_dim
        self.expansion = expansion # expansion = out_dim // in_dim, for ds block, should have multuiple expansion
        self.hidden_dim = in_dim*self.expansion
        self.out_dim = in_dim*self.expansion
        self.vector_dim = self.out_dim // 1
        self.n_sample = n_sample
        self.fps_rate = fps_rate # if apply fps rate, use the FPS to downsample the points first

        # whether to use the vector att/original attention
        self.use_vector_attn = True
        if not self.use_vector_attn:
            self.nhead = 4

        self.linear_top = nn.Sequential(
            nn.Conv1d(in_dim, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim)
        )
        self.linear_down = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.out_dim, 1),
            nn.BatchNorm1d(self.out_dim)
        )

        self.phi = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            # nn.BatchNorm1d(self.out_dim) 
        )
        self.psi = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            # nn.BatchNorm1d(self.out_dim) 
        )

        if self.fps_rate is not None:
            # downsample_block cfg
            self.POS_ENCODING=True
            self.CAT_POS=False
            self.SKIP_ATTN = True # skip-attn means linear + max(mini-pointnet)

            self.QK_POS_ONLY = False
            self.V_POS_ONLY = False
            self.MAX_POOL = False

            self.SKIP_ALL = False # only fps
            self.USE_KNN = False
        else:
            # normal block cfg
            # DEBUG: if the normal block has skip-attn will result in nan
            self.POS_ENCODING=True
            self.CAT_POS=False
            self.SKIP_ATTN = True

            self.QK_POS_ONLY = False
            self.V_POS_ONLY = False
            self.MAX_POOL = False

            self.SKIP_ALL = False
            self.USE_KNN = False

        if not self.USE_KNN:
            assert radius is not None
            self.radius = radius

        if self.SKIP_ALL:
            self.tmp_linear = nn.Sequential(
                    nn.Conv1d(self.in_dim, self.out_dim,1),
                    nn.BatchNorm1d(self.out_dim),
                    nn.ReLU(),
                    )

        if self.SKIP_ATTN:
            self.alpha = nn.Sequential(
                    nn.Conv2d(self.in_dim+3, self.in_dim, 1) if self.CAT_POS else nn.Conv2d(self.in_dim, self.in_dim, 1),
                    nn.BatchNorm2d(self.in_dim),
                    nn.ReLU(),
                    nn.Conv2d(self.in_dim, self.hidden_dim, 1),
                    nn.BatchNorm2d(self.hidden_dim),
                    nn.ReLU(),
                    )
        else:
            self.alpha = nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
                # nn.BatchNorm1d(self.hidden_dim)
            )

        self.gamma = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.vector_dim, 1),
            nn.BatchNorm2d(self.vector_dim),
        )

        if self.SKIP_ATTN:
            # skip-attn, pos_encoing should out in_dim
            self.delta = nn.Sequential(
                nn.Conv2d(3, self.in_dim, 1),
                nn.BatchNorm2d(self.in_dim),
                nn.ReLU(),
                nn.Conv2d(self.in_dim, self.in_dim, 1),
                nn.BatchNorm2d(self.in_dim),
                # nn.ReLU(),
                )
        else:
            self.delta = nn.Sequential(
                nn.Conv2d(3, self.hidden_dim, 1),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
                nn.BatchNorm2d(self.hidden_dim),
                # nn.ReLU(),
                )

    def forward(self, input_p, input_x):
        '''
        input_p:  B, 3, npoint
        input_x: B, in_dim, npoint
        '''
        B, in_dim, npoint = list(input_x.size()) # npoint: the input point-num
        n_sample = self.n_sample       # the knn-sample num cur block
        k = min(n_sample, npoint)      # denoting the num_point cur layer
        if not self.use_vector_attn:
            h = self.nhead                  # only used in non-vextor attn

        input_p = input_p.permute([0,2,1]) # [B, npoint, 3]
        self.register_buffer('in_xyz_map', input_p)

        if self.fps_rate is not None:
            npoint = npoint // self.fps_rate
            fps_idx = farthest_point_sample_cuda(input_p, npoint)
            torch.cuda.empty_cache()
            input_p_fps = index_points_cuda(input_p, fps_idx) # [B. M, 3]
            if self.SKIP_ALL:
                input_p_reduced = input_p_fps.transpose(1,2)
                input_x_reduced = index_points_cuda(self.tmp_linear(input_x).transpose(1,2), fps_idx).transpose(1,2)
                return input_p_reduced, input_x_reduced
        else:
            input_p_fps = input_p
            input_x_fps = input_x

        res = input_x # [B, dim, M]

        if self.USE_KNN:
            self.knn = KNN(k=k, transpose_mode=True)
            _, idx = self.knn(input_p.contiguous(), input_p_fps.contiguous())
            idx = idx.int() # [bs, npoint, k]
        else:
            idx = query_ball_point_cuda(self.radius, k, input_p.contiguous(), input_p_fps.contiguous()) # [bs, npoint, k]

        grouped_input_p = grouping_operation_cuda(input_p.transpose(1,2).contiguous(), idx) # [bs, xyz, npoint, k]
        grouped_input_x = grouping_operation_cuda(input_x.contiguous(), idx) # [bs, hidden_dim, npoint, k]

        self.register_buffer('neighbor_map', idx)

        # TODO: define proper r for em
        # query_idx = query_ball_point_cuda(radius, k, coord, coord) # [bs, npoint, k]
        # self.knn = KNN(k=k, transpose_mode=True)
        # _, knn_idx = self.knn(input_p.contiguous(), input_p)
        # import ipdb; ipdb.set_trace()

        if self.fps_rate is not None:
            if self.SKIP_ATTN:
                pass # only apply linear-top for ds blocks
            else:
                input_x = self.linear_top(input_x)
        else:
            if self.SKIP_ATTN:
                pass # only apply linear-top for ds blocks
            else:
                input_x = self.linear_top(input_x)
        # input_x = self.linear_top(input_x)

        if self.SKIP_ATTN:
            # import ipdb; ipdb.set_trace()
            # out_dim should be the same with in_dim, since here contains no TD
            if self.POS_ENCODING:
                relative_xyz = input_p_fps.permute([0,2,1])[:,:,:,None] - grouped_input_p
                pos_encoding = self.delta(relative_xyz)    # [bs, dims, npoint, k]
                if self.CAT_POS:
                    alpha = self.alpha(torch.cat([grouped_input_x, relative_xyz], dim=1))
                else: # use sum 
                    alpha = self.alpha(grouped_input_x + pos_encoding)
            else:
                alpha = self.alpha(grouped_input_x)
                # alpha = grouping_operation_cuda(self.alpha(input_x).contiguous(), idx)

            y = alpha.max(dim=-1)[0]
            # y = alpha.sum(dim=-1)
            y = self.linear_down(y)

            if self.fps_rate is not None:
                input_p_reduced = input_p_fps.transpose(1,2)
                # WRONG!: noneed for applying fps_idx here
                # input_x_reduced = index_points_cuda(y.transpose(1,2), fps_idx).transpose(1,2)  # [B, dim, M]
                input_x_reduced = y
                return input_p_reduced, input_x_reduced
            else:
                input_p_reduced = input_p_fps.transpose(1,2)
                input_x_reduced = y + res
                return input_p_reduced, input_x_reduced

        # when downsampling the TRBlock
        # should use downsampled qkv here, so use input_x_fps
        # as for normal block, input_x and input_x_fps are the same
        if self.fps_rate is not None:
            input_x_fps = index_points_cuda(input_x.transpose(1,2), fps_idx).transpose(1,2)  # it is only used for tr-like downsample block
            phi = self.phi(input_x_fps)
        else:
            phi = self.phi(input_x)
        phi = phi[:,:,:,None].repeat(1,1,1,k)
        psi = grouping_operation_cuda(self.psi(input_x).contiguous(), idx)
        self.skip_knn = True
        alpha = grouping_operation_cuda(self.alpha(input_x).contiguous(), idx) # [bs, xyz, npoint, k]

        if self.POS_ENCODING:
            relative_xyz = input_p_fps.permute([0,2,1])[:,:,:,None] - grouped_input_p
            pos_encoding = self.delta(relative_xyz)    # [bs, dims, npoint, k]

        if self.use_vector_attn:
            # the attn_map: [vector_dim];
            # the alpha:    [out_dim]
            if self.POS_ENCODING:
                # if V_POS and QK_POS is both false, then apply all pos_encoding 
                assert (self.V_POS_ONLY and self.QK_POS_ONLY) is False  # only one of the V_ONLY and QK_ONLY should be applied
                if self.V_POS_ONLY:
                    attn_map = F.softmax(self.gamma(phi - psi), dim=-1)
                else:
                    attn_map = F.softmax(self.gamma(phi - psi + pos_encoding), dim=-1)
                if self.QK_POS_ONLY:
                    y = attn_map.repeat(1, self.out_dim // self.vector_dim,1,1)*(alpha)
                else:
                    y = attn_map.repeat(1, self.out_dim // self.vector_dim,1,1)*(alpha+pos_encoding)
            else:
                attn_map = F.softmax(self.gamma(phi - psi), dim=-1)
                y = attn_map.repeat(1, self.out_dim // self.vector_dim,1,1)*(alpha)
            if self.MAX_POOL:
                y = y.max(dim=-1)[0]
            else:
                y = y.sum(dim=-1)
        else:
            assert self.POS_ENCODING == True
            phi = phi.reshape(B, h, self.out_dim//h, npoint, k)
            psi = psi.reshape(B, h, self.out_dim//h, npoint, k)
            attn_map = F.softmax((phi*psi).reshape(B, self.out_dim, npoint, k) + pos_encoding, dim=-1)
            y = attn_map*(alpha+pos_encoding)
            y = y.sum(dim=-1)

        self.register_buffer('attn_map', attn_map.mean(dim=1))

        y = self.linear_down(y)

        if self.fps_rate is not None:
            input_p_reduced = input_p_fps.transpose(1,2)
            # input_x_reduced = index_points_cuda(y.transpose(1,2), fps_idx).transpose(1,2)  # [B, dim, M]
            input_x_reduced = y
            return input_p_reduced, input_x_reduced
        else:
            input_p_reduced = input_p_fps.transpose(1,2)
            input_x_reduced = y + res
            return input_p_reduced, input_x_reduced

