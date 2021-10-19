import os
import shlex
import subprocess

import h5py
import numpy as np
import torch
import torch.utils.data as data

from lib.sparse_voxelization import SparseVoxelizer

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]


def _load_data_file(name):
    f = h5py.File(name, "r")
    data = f["data"][:]
    label = f["label"][:]
    return data, label


class S3DIS(data.Dataset):
    def __init__(self, config, train=True, download=False, data_precent=1.0):
        super().__init__()
        self.data_precent = data_precent
        self.folder = "indoor3d_sem_seg_hdf5_data"
        self.data_path = "/data/eva_share_users/zhaotianchen/"
        self.data_dir = os.path.join(self.data_path, self.folder)
        self.url = (
            "https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip"
        )
        self.train = train
        self.split = "train" if self.train else "val"

        self.NUM_IN_CHANNEL=9
        self.NUM_LABELS=13
        self.IGNORE_LABELS = [-1]  # labels that are not evaluated
        self.NEED_PRED_POSTPROCESSING = False

        if self.train:
            self.data_dict = torch.load(os.path.join(self.data_dir, "s3dis_train.pth"),'cpu')
        else:
            self.data_dict = torch.load(os.path.join(self.data_dir, "s3dis_val.pth"), 'cpu')

        # self.data_dict = torch.load(os.path.join(self.data_dir, "s3dis_debug.pth"), 'cpu')

        # DEBUG: dirty fixing
        self.data_dict['data'] = np.concatenate(self.data_dict['data'], axis=0)    # [batches, 4096, 9]
        self.data_dict['label'] = np.concatenate(self.data_dict['label'], axis=0)    # [batches, 4096, 9]

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

        # if download and not os.path.exists(self.data_dir):
            # zipfile = os.path.join(self.data_dir, os.path.basename(self.url))
            # subprocess.check_call(
                # shlex.split("curl {} -o {}".format(self.url, zipfile))
            # )

            # subprocess.check_call(
                # shlex.split("unzip {} -d {}".format(zipfile, self.data_dir))
            # )

            # subprocess.check_call(shlex.split("rm {}".format(zipfile)))


        # all_files = _get_data_files(os.path.join(self.data_dir, "all_files.txt"))
        # room_filelist = _get_data_files(
            # os.path.join(self.data_dir, "room_filelist.txt")
        # )

        # data_batchlist, label_batchlist = [], []
        # for f in all_files:
            # data, label = _load_data_file(os.path.join(self.data_path, f))
            # data_batchlist.append(data)
            # label_batchlist.append(label)

        # data_batches = np.concatenate(data_batchlist, 0)
        # labels_batches = np.concatenate(label_batchlist, 0)

        # test_area = "Area_5"
        # train_idxs, test_idxs = [], []

        # d = {}
        # d_val = {}
        # for i,room_name in enumerate(room_filelist):
            # if room_name not in d.keys():
                # d[room_name] = []
            # d[room_name].append(i)

        # test_keys = []
        # for k in d.keys():
            # if test_area in k:
                # test_keys.append(k)

        # for k in test_keys:
            # d_val[k] = d.pop(k)

        # train_names = d.keys()
        # val_names = d_val.keys()

        # d2save_train = {}
        # d2save_train['data'] = []
        # d2save_train['label'] = []
        # d2save_train['scene_name'] = list(train_names)
        # d2save_val = {}
        # d2save_val['data'] = []
        # d2save_val['label'] = []
        # d2save_val['scene_name'] = list(val_names)

        # for _, idx in d.items():
            # d2save_train['data'].append(data_batches[idx])
            # d2save_train['label'].append(labels_batches[idx])

        # for _, idx in d_val.items():
            # d2save_val['data'].append(data_batches[idx])
            # d2save_val['label'].append(labels_batches[idx])

        # ======================================================

        # # if self.train:
            # # self.points = data_batches[train_idxs, ...]
            # # self.labels = labels_batches[train_idxs, ...]
        # # else:
            # # self.points = data_batches[test_idxs, ...]
            # # self.labels = labels_batches[test_idxs, ...]

        # torch.save(d2save_train, os.path.join(self.data_dir, 's3dis_train.pth'))
        # torch.save(d2save_val, os.path.join(self.data_dir, 's3dis_val.pth'))
        # d_debug = {}
        # for k in d2save_train.keys():
            # d_debug[k] = d2save_train[k][:3]
        # torch.save(d_debug, os.path.join(self.data_dir, 's3dis_debug.pth'))

    def __getitem__(self, index):

        data = self.data_dict['data'][index].reshape([-1,9])
        target = self.data_dict['label'][index].reshape([-1])
        data_ = data

        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            data[:, :3] = np.dot(data_[:, :3], rot_mat) * scale_factor
        else:
            pass

        coords = data[:,:3] - (data[:,:3]).min(0)
        outs = self.sparse_voxelizer.voxelize(
            coords, # debug, not sure
            data_,
            target,
            center=None,
            rotation_angle=None,
            return_transformation=False
            )

        # d = {}
        # d['label'] = target
        # d['origin_pc'] = data_[:,:3]
        # d['v_coord'] = outs[0]
        # d['v_label'] = outs[2]
        # torch.save(d, "./plot/data-s3dis.pth")
        # import ipdb; ipdb.set_trace()

        # outs = (coords, feats, labels, unique_map, inverse_map)
        assert isinstance(outs, tuple)
        return outs

    def __len__(self):
        return len(self.data_dict['data'])

    def cleanup(self):
        self.sparse_voxelizer.cleanup()

    def reorder_result(self, result):
        return result

if __name__ == "__main__":
    dset = Indoor3DSemSeg(16)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())
