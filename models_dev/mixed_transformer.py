import torch
import torch.nn as nn
import torch.nn.functional as F

import MinkowskiEngine as ME
from models.pct_utils import TDLayer, TULayer, PTBlock
from models.pct_voxel_utils import separate_batch, voxel2points, points2voxel

class MixedTransformer(nn.Module):
    def __init__(self,config,num_class,N,normal_channel=3):
        super(MixedTransformer, self).__init__()
        # The normal channel for Modelnet is 3, for scannet is 6, for scanobjnn is 0
        in_channel = normal_channel+3 # normal ch + xyz
        self.normal_channel = normal_channel

        self.input_mlp = nn.Sequential(
            nn.Conv1d(in_channel, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32))

        self.in_dims = [32, 64, 128, 256]
        self.out_dims = [64, 128, 256, 512]
        self.neighbor_ks = [16, 16, 16, 16, 16]

        self.PTBlock0 = PTBlock(in_dim=self.in_dims[0], n_sample=self.neighbor_ks[0])

        # self.TDLayer1 = TDLayer(npoint=int(N/4),input_dim=self.in_dims[0], out_dim=self.out_dims[0], k=self.neighbor_ks[1])
        self.TDLayer1 = PTBlock(in_dim=self.in_dims[0], n_sample=self.neighbor_ks[1], fps_rate = 2, expansion=2)
        self.PTBlock1 = PTBlock(in_dim=self.out_dims[0], n_sample=self.neighbor_ks[1])

        # self.TDLayer2 = TDLayer(npoint=int(N/16),input_dim=self.in_dims[1], out_dim=self.out_dims[1], k=self.neighbor_ks[2])
        self.TDLayer2 = PTBlock(in_dim=self.in_dims[1], n_sample=self.neighbor_ks[2], fps_rate = 2, expansion=2)
        self.PTBlock2 = PTBlock(in_dim=self.out_dims[1], n_sample=self.neighbor_ks[2])

        # self.TDLayer3 = TDLayer(npoint=int(N/64),input_dim=self.in_dims[2], out_dim=self.out_dims[2], k=self.neighbor_ks[3])
        self.TDLayer3 = PTBlock(in_dim=self.in_dims[2], n_sample=self.neighbor_ks[3], fps_rate = 2, expansion=2)
        self.PTBlock3 = PTBlock(in_dim=self.out_dims[2], n_sample=self.neighbor_ks[3])

        # self.TDLayer4 = TDLayer(npoint=int(N/256),input_dim=self.in_dims[3], out_dim=self.out_dims[3], k=self.neighbor_ks[4])
        self.TDLayer4 = PTBlock(in_dim=self.in_dims[3], n_sample=self.neighbor_ks[4], fps_rate = 2, expansion=2)
        self.PTBlock4 = PTBlock(in_dim=self.out_dims[3], n_sample=self.neighbor_ks[4])

        self.middle_linear = nn.Conv1d(self.out_dims[3], self.out_dims[3],1)
        self.PTBlock_middle = PTBlock(in_dim=self.out_dims[3], n_sample=self.neighbor_ks[4])

        self.TULayer5 = TULayer(npoint=int(N/64),input_dim=self.out_dims[3], out_dim=self.in_dims[3], k=3)
        self.PTBlock5= PTBlock(in_dim=self.in_dims[3], n_sample=self.neighbor_ks[4])

        self.TULayer6 = TULayer(npoint=int(N/16),input_dim=self.out_dims[2], out_dim=self.in_dims[2], k=3)
        self.PTBlock6= PTBlock(in_dim=self.in_dims[2], n_sample=self.neighbor_ks[3])

        self.TULayer7 = TULayer(npoint=int(N/4),input_dim=self.out_dims[1], out_dim=self.in_dims[1], k=3)
        self.PTBlock7= PTBlock(in_dim=self.in_dims[1], n_sample=self.neighbor_ks[2])

        self.TULayer8 = TULayer(npoint=int(N),input_dim=self.out_dims[0], out_dim=self.in_dims[0], k=3)
        self.PTBlock8= PTBlock(in_dim=self.in_dims[0], n_sample=self.neighbor_ks[1])

        self.fc = nn.Sequential(
            nn.Linear(32,32),
            nn.Dropout(0.4),
            nn.Linear(32,num_class),
        )

        self.use_ln = False
        if self.use_ln:
            self.final_ln = nn.LayerNorm(256)

        self.save_flag = False
        self.save_dict = {}
        for i in range(5):
            self.save_dict['attn_{}'.format(i)] = []

        self.conv1 = nn.Conv1d(self.in_dims[0], self.out_dims[0], 1)
        self.conv2 = nn.Conv1d(self.in_dims[1], self.out_dims[1], 1)
        self.conv3 = nn.Conv1d(self.in_dims[2], self.out_dims[2], 1)
        self.conv4 = nn.Conv1d(self.in_dims[3], self.out_dims[3], 1)
        self.conv5 = nn.Conv1d(self.out_dims[3], num_class, 1)

    def save_intermediate(self):

        save_dict = self.save_dict
        self.save_dict = {}
        for i in range(5):
            self.save_dict['attn_{}'.format(i)] = []
        return save_dict


    def forward(self, inputs):

        xyz, points, idx = voxel2points(inputs)
        inputs_ = points.transpose(1,2)

        if self.normal_channel:
            l0_xyz = inputs_[:, :3, :]
        else:
            l0_xyz = inputs_

        input_points = self.input_mlp(inputs_)

        l0_points, attn_0 = self.PTBlock0(l0_xyz, input_points)

        l1_xyz, l1_points = self.TDLayer1(l0_xyz, l0_points)
        l1_points, attn_1 = self.PTBlock1(l1_xyz, l1_points)

        l2_xyz, l2_points = self.TDLayer2(l1_xyz, l1_points)
        l2_points, attn_2 = self.PTBlock2(l2_xyz, l2_points)

        l3_xyz, l3_points = self.TDLayer3(l2_xyz, l2_points)
        l3_points, attn_3 = self.PTBlock3(l3_xyz, l3_points)

        l4_xyz, l4_points = self.TDLayer4(l3_xyz, l3_points)
        l4_points, attn_4 = self.PTBlock4(l4_xyz, l4_points)

        l4_points = self.middle_linear(l4_points)
        l4_points, attn_4 = self.PTBlock_middle(l4_xyz, l4_points)

        l5_xyz, l5_points = self.TULayer5(l4_xyz, l3_xyz, l4_points, l3_points)
        l5_points, attn_5 = self.PTBlock5(l5_xyz, l5_points)

        l6_xyz, l6_points = self.TULayer6(l5_xyz, l2_xyz, l5_points, l2_points)
        l6_points, attn_6 = self.PTBlock6(l6_xyz, l6_points)

        l7_xyz, l7_points = self.TULayer7(l6_xyz, l1_xyz, l6_points, l1_points)
        l7_points, attn_7 = self.PTBlock7(l7_xyz, l7_points)

        l8_xyz, l8_points = self.TULayer8(l7_xyz, l0_xyz, l7_points, l0_points)
        l8_points, attn_8 = self.PTBlock8(l8_xyz, l8_points)

        x = self.fc(l8_points.transpose(1,2))

        num_class = x.shape[-1]

        new_x = points2voxel(x, idx)
        y = ME.SparseTensor(features=new_x, coordinates=inputs.C, coordinate_manager=inputs.coordinate_manager)

        if torch.isinf(x).sum() > 0:
            import ipdb; ipdb.set_trace()

        return y


class get_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, predict, target, weights):
        """
        :param predict: (B,N,C)
        :param target: (B,N)
        :param weights: (B,N)
        :return:
        """
        NUM_CLASSES=20
        predict = predict.view(-1, NUM_CLASSES).contiguous() # B*N, C
        target = target.view(-1).contiguous().cuda().long()  # B*N
        weights = weights.view(-1).contiguous().cuda().float() # B*N

        loss = self.cross_entropy_loss(predict, target) # B*N
        loss *= weights
        loss = torch.mean(loss)
        return loss

