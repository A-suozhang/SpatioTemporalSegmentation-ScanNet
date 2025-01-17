import os
import os.path
import torch

import numpy as np
import torch.nn.functional as F

from lib.sparse_voxelization import SparseVoxelizer

__all__ = ['SemanticKITTI']

label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


class Remap(object):
    def __init__(self) -> None:
        kept_labels_ordered = []
        for k in label_name_mapping:
            _label = label_name_mapping[k]
            if _label in kept_labels:
                kept_labels_ordered.append(_label)

        inv_map = dict()
        for i, label in enumerate(kept_labels_ordered):
            key = i
            for k in label_name_mapping:
                if label_name_mapping[k] == label:
                    value = k
                    break
            inv_map[key] = value

        max_key = max(inv_map.keys()) + 1
        remap_lut = np.zeros((max_key), dtype=np.int32)
        remap_lut[list(inv_map.keys())] = list(inv_map.values())
        self.remap_lut = remap_lut
    
    def getRemapLUT(self):
        return self.remap_lut


class SemanticKITTI(dict):

    def __init__(self, root, voxel_size, num_points,**kwargs):

        submit_to_server = kwargs.get('submit', False)
        sample_stride = kwargs.get('sample_stride', 1)
        google_mode = kwargs.get('google_mode', False)

        if submit_to_server:
            super().__init__({
                'train':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          split='train',
                                          num_points=None,
                                          sample_stride=1,
                                          submit=True),
                'test':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          split='test',
                                          # split='val',   # DEBUG_ONLY
                                          num_points=num_points,
                                          sample_stride=1,
                                          submit=True)
            })
        else:
            super().__init__({
                'train':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          split='train',
                                          num_points=None,
                                          sample_stride=sample_stride,
                                          google_mode=google_mode),
                'test':
                    SemanticKITTIInternal(root,
                                          voxel_size,
                                          split='val',
                                          num_points=None,
                                          sample_stride=sample_stride
                    )
            })


class SemanticKITTIInternal:

    def __init__(self,
                 root,
                 voxel_size,
                 split,
                 num_points=None,
                 sample_stride=1,
                 submit=False,
                 google_mode=True):
        if submit:
            trainval = True
        else:
            trainval = False
        self.submit = submit
        self.NUM_IN_CHANNEL = 4
        self.NUM_LABELS = len(kept_labels)
        self.IGNORE_LABELS = [-1]  # labels that are not evaluated
        self.NEED_PRED_POSTPROCESSING = True
        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.sample_stride = sample_stride
        self.google_mode = google_mode
        self.seqs = []
        if split == 'train':
            self.seqs = [
                '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
            ]
            if self.google_mode or trainval:
                self.seqs.append('08')
        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = [
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
            ]
        self.files = []
        for seq in self.seqs:
            seq_files = sorted(
                os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [
                os.path.join(self.root, seq, 'velodyne', x) for x in seq_files
            ]
            self.files.extend(seq_files)

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        # Perform label mapping
        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-',
                                                        '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[
                        label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = -1
            elif label_id == 0:
                self.label_map[label_id] = -1
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[
                        label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = -1

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = cnt
        self.angle = 0.0

        self.class_names = kept_labels

        # We don't do any more augmentation in SparseVoxelizer
        # because we already done in this Class
        self.sparse_voxelizer = SparseVoxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=False,
            scale_augmentation_bound=None,
            rotation_augmentation_bound=None,
            translation_augmentation_ratio_bound=None,
            rotation_axis=0, # this isn't actually used
            ignore_label=-1
        )

        # self.split = 'val'
        # outs = self.__getitem__(0)
        # d1 = {}
        # d1['coords'] = outs[0]
        # d1['feats'] = outs[1]
        # d1['target'] = outs[2]
        # torch.save(d1, '/home/zhaotianchen/project/point-transformer/SpatioTemporalSegmentation-ScanNet/plot/kitti-my.pth')
        self.use_class_reweight = False
        if self.use_class_reweight:
            self.point_num_ratio = torch.tensor(torch.load('./point_ratio_val.pth')).cuda()
            self.class_reweight_lambda = 1.e-1 # around 3x
            self.class_reweight_topk = 1

            if self.class_reweight_lambda:
                class_reweight_masks = self.point_num_ratio.argsort()[self.class_reweight_topk:]
            else:
                class_reweight_masks = []
                # class_reweight_indexes = [7,11] 

            class_reweight = F.softmax(self.point_num_ratio/self.class_reweight_lambda)*len(self.point_num_ratio)
            class_reweight[class_reweight_masks] = 1
            class_reweight = 1/class_reweight
            self.class_reweight = class_reweight


    def get_prediction(self, output, target):
        if not self.use_class_reweight:
            return output.max(1)[1]
        else:
            output_ = F.softmax(output/5)  # make the pred softer, simialr with 0.1
            output_ = output_*self.class_reweight
            # print(class_reweight)

            # debugging the 7,11 class missing
            debug_index = 7
            if len(torch.where(target == debug_index)[0])>0:
                idxs = torch.where(target == debug_index)[0]
                for idx_ in idxs:
                    print(output_[idx_].max(), output_[idx_][debug_index])
                print('\n')

            return output_.max(1)[1]

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def get_classnames(self):
        kept_labels_ordered = []
        for k in label_name_mapping:
            _label = label_name_mapping[k]
            if _label in kept_labels:
                kept_labels_ordered.append(_label)

        return kept_labels_ordered


    def __getitem__(self, index):
        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        block = np.zeros_like(block_) # block is input data

        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3] # make sure we have the unaltered remission data
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32) # pc is xyz coordinate data
        pc_ -= pc_.min(0, keepdims=1)

        label_file = self.files[index].replace('velodyne', 'labels').replace(
            '.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
        else:
            all_labels = np.zeros(pc_.shape[0]).astype(np.int32)

        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)

        feat_ = block # feature is [x, y, z, remission]

        if 'train' in self.split:
            if self.num_points is not None and len(inds) > self.num_points:
                print('exceed num-points, subsampling...')
                inds = np.random.choice(inds, self.num_points, replace=False)
        new_c = block[:,:3] - (block[:,:3]).min(0)
        # new_c = np.round(pc_[:,:3]*self.voxel_size)

        outs = self.sparse_voxelizer.voxelize(
            new_c, # debug, not sure
            # pc_[:,:3],
            feat_,
            labels_,
            center=None,
            rotation_angle=None,
            return_transformation=False
        )

        # add filename to save result for submission
        outs = (*outs, self.files[index])
        # outs = (coords, feats, labels, unique_map, inverse_map)
        assert isinstance(outs, tuple)
        return outs

    def cleanup(self):
        self.sparse_voxelizer.cleanup()
    
    def reorder_result(self, result):
        return result
