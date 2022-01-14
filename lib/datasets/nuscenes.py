import os
import shlex
import subprocess

import h5py
import pickle
import yaml
import numpy as np
import numba as nb
import torch
import torch.utils.data as data

from lib.sparse_voxelization import SparseVoxelizer
from nuscenes import NuScenes

def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


class Nuscenes(data.Dataset):
    def __init__(self, config, train=True, cylinder_voxelize=False, sample_stride=1):

        self.config = config
        self.train = train
        self.split='train' if self.train else "val"
        self.cylinder_voxelize = cylinder_voxelize
        self.sample_stride = sample_stride

        self.NUM_IN_CHANNEL=5
        self.NUM_LABELS=16 # TODO: fix
        self.IGNORE_LABELS = [-1]  # DEBUG: not actually used
        self.NEED_PRED_POSTPROCESSING = False

        self.metadata_pickle_path="/data/eva_share_users/zhaotianchen/nuscenes/nuscenes_infos_{}.pkl".format(self.split)
        self.label_mapping_path="/data/eva_share_users/zhaotianchen/nuscenes/nuscenes_label_mapping.yaml"
        self.label_filename_path="/data/eva_share_users/zhaotianchen/nuscenes/nuscenes_label_filename_{}.pkl".format(self.split)
        self.data_path="/data/eva_share_users/zhaotianchen/nuscenes"

    # def __init__(self, data_path, imageset='train',
                 # return_ref=False, label_mapping="nuscenes.yaml", nusc=None):
        # self.return_ref = return_ref

        # load a few elements
        with open(self.metadata_pickle_path, 'rb') as f:
            metadata = pickle.load(f)
        with open(self.label_mapping_path, 'r') as stream:
            label_mapping = yaml.safe_load(stream)
        self.learning_map = label_mapping['learning_map']
        self.nusc_infos = metadata['infos']

        # TODO: instead of loading the whole nusc pickle file, just load the label filenames
        # self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_path, verbose=True)
        # with open("/data/eva_share_users/zhaotianchen/nuscenes/nusc.pkl", 'rb') as f:
            # self.nusc = pickle.load(f)
        # self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_path, verbose=True)
        with open(self.label_filename_path, 'rb') as f:
            self.label_filenames = pickle.load(f)

        self.nusc_infos = self.nusc_infos[::self.sample_stride]
        self.label_filenames = self.label_filenames[::self.sample_stride]

        # self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray([480, 360, 32])
        self.rotate_aug = False
        self.flip_aug = False
        self.scale_aug = False
        self.ignore_label = 0
        self.return_test = False
        self.fixed_volume_space = False
        self.max_volume_space = [50, np.pi, 3]
        self.min_volume_space = [0, -np.pi, -5]
        self.transform = False
        self.trans_std = [0.1, 0.1, 0.1]

        min_rad = -np.pi/4
        max_rad = np.pi/4
        self.noise_rotation = np.random.uniform(min_rad, max_rad)

        self.sparse_voxelizer = SparseVoxelizer(
                voxel_size=config.voxel_size,
                clip_bound=None,
                use_augmentation=False,
                scale_augmentation_bound=None,
                rotation_augmentation_bound=None,
                translation_augmentation_ratio_bound=None,
                rotation_axis=0, # this isn't actually used
                ignore_label=-1
            )

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos) 

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        # lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        # lidarseg_labels_filename = os.path.join(self.data_path,
                                                # self.nusc.get('lidarseg', lidar_sd_token)['filename'])
        lidarseg_labels_filename = os.path.join(self.data_path, self.label_filenames[index])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        # data_tuple = (points[:, :3], points_label.astype(np.uint8))
        # if self.return_ref:
            # data_tuple += (points[:, 3],)

        # ------ split of 2 files: pc_dataset & dataset_nuscenes.py -------

        # data = self.point_cloud_dataset[index]
        # if len(data) == 2:
            # xyz, labels = data
        # elif len(data) == 3:
            # xyz, labels, sig = data
            # if len(sig.shape) == 2: sig = np.squeeze(sig)
        # else:
            # raise Exception('Return invalid data tuple

        xyz = points[:,:3]
        labels = np.squeeze(points_label - 1, axis=-1)  # INFO: the label is 0~16, minus one adnd ignore -1
        points[:,:3] = (points[:,:3] / points[:,:3].max() - 0.5) # normalize the coordinate xyz in feature

        if not self.cylinder_voxelize:
            xyz[:,:3] -= np.expand_dims(xyz[:,:3].min(0), axis=0)
            outs = self.sparse_voxelizer.voxelize(
                    xyz,
                    points,
                    labels,
                    center=None,
                    rotation_angle=None,
                    return_transformation=False
                    )

            # for visualization:
            # d = {}
            # d['coord_pt'] = xyz
            # d['label_pt'] = labels
            # d['coord_voxel'] = outs[0]
            # d['label_voxel'] = outs[2]
            # torch.save(d, './nuscene-voxel-demo.pth')
            # import ipdb; ipdb.set_trace()
            return outs # (coord, feat, target)


        else:  # conduct cylinder-like voxelization
            # TODO: cylinder voxelization needs extra network part to align, hard to adapt, use naive way

            # random data augmentation by rotation
            if self.rotate_aug:
                rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
                c, s = np.cos(rotate_rad), np.sin(rotate_rad)
                j = np.matrix([[c, s], [-s, c]])
                xyz[:, :2] = np.dot(xyz[:, :2], j)

            # random data augmentation by flip x , y or x+y
            if self.flip_aug:
                flip_type = np.random.choice(4, 1)
                if flip_type == 1:
                    xyz[:, 0] = -xyz[:, 0]
                elif flip_type == 2:
                    xyz[:, 1] = -xyz[:, 1]
                elif flip_type == 3:
                    xyz[:, :2] = -xyz[:, :2]
            if self.scale_aug:
                noise_scale = np.random.uniform(0.95, 1.05)
                xyz[:, 0] = noise_scale * xyz[:, 0]
                xyz[:, 1] = noise_scale * xyz[:, 1]
            # convert coordinate into polar coordinates

            if self.transform:
                noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                            np.random.normal(0, self.trans_std[1], 1),
                                            np.random.normal(0, self.trans_std[2], 1)]).T

                xyz[:, 0:3] += noise_translate

            xyz_pol = cart2polar(xyz)

            max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
            min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
            max_bound = np.max(xyz_pol[:, 1:], axis=0)
            min_bound = np.min(xyz_pol[:, 1:], axis=0)
            max_bound = np.concatenate(([max_bound_r], max_bound))
            min_bound = np.concatenate(([min_bound_r], min_bound))
            if self.fixed_volume_space:
                max_bound = np.asarray(self.max_volume_space)
                min_bound = np.asarray(self.min_volume_space)
            # get grid index
            crop_range = max_bound - min_bound
            cur_grid_size = self.grid_size
            intervals = crop_range / (cur_grid_size - 1)

            if (intervals == 0).any(): print("Zero interval!")
            grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

            voxel_position = np.zeros(self.grid_size, dtype=np.float32)
            dim_array = np.ones(len(self.grid_size) + 1, int)
            dim_array[0] = -1
            voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
            voxel_position = polar2cat(voxel_position)

            # process labels
            processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
            label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
            label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
            processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
            data_tuple = (voxel_position, processed_label)

            # center data on each voxel for PTnet
            voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
            return_xyz = xyz_pol - voxel_centers
            return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

            # if len(data) == 2:
                # return_fea = return_xyz
            # elif len(data) == 3:
                # return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)
            return_fea = return_xyz

            if self.return_test:
                data_tuple += (grid_ind, labels, return_fea, index)
            else:
                data_tuple += (grid_ind, labels, return_fea)

            d1 = {
                'coord': grid_ind,
                'label': labels,
            }
            torch.save(d1, './nuscene-cylinder-voxel.pth')

            import ipdb; ipdb.set_trace()

            return data_tuple



    # def __len__(self):
        # if not self.whole_scene:
            # return len(self.data_dict['data'])
        # else:
            # return len(self.filelists)

    def cleanup(self):
        self.sparse_voxelizer.cleanup()

    def reorder_result(self, result):  # used for valid
        return result


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def collate_fn_BEV(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz

if __name__ == "__main__":
    config_ = {}
    dset = Nuscenes(config_)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())
