# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import os
import glob
from abc import ABC
from tqdm import tqdm
import os.path as osp
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
from enum import Enum

from lib.sparse_voxelization import SparseVoxelizer

from torch.utils.data import Dataset, DataLoader

from lib.pc_utils import read_plyfile
import lib.transforms as t
from lib.dataloader import InfSampler
import MinkowskiEngine as ME

import torch


class DatasetPhase(Enum):
  Train = 0
  Val = 1
  Val2 = 2
  TrainVal = 3
  Test = 4


def datasetphase_2str(arg):
  if arg == DatasetPhase.Train:
    return 'train'
  elif arg == DatasetPhase.Val:
    return 'val'
  elif arg == DatasetPhase.Val2:
    return 'val2'
  elif arg == DatasetPhase.TrainVal:
    return 'trainval'
  elif arg == DatasetPhase.Test:
    return 'test'
  else:
    raise ValueError('phase must be one of dataset enum.')


def str2datasetphase_type(arg):
  if arg.upper() == 'TRAIN':
    return DatasetPhase.Train
  elif arg.upper() == 'VAL':
    return DatasetPhase.Val
  elif arg.upper() == 'VAL2':
    return DatasetPhase.Val2
  elif arg.upper() == 'TRAINVAL':
    return DatasetPhase.TrainVal
  elif arg.upper() == 'TEST':
    return DatasetPhase.Test
  else:
    raise ValueError('phase must be one of train/val/test')


def cache(func):

  def wrapper(self, *args, **kwargs):
    # Assume that args[0] is index
    index = args[0]
    if self.cache:
      if index not in self.cache_dict[func.__name__]:
        results = func(self, *args, **kwargs)
        self.cache_dict[func.__name__][index] = results
      return self.cache_dict[func.__name__][index]
    else:
      return func(self, *args, **kwargs)

  return wrapper


class DictDataset(Dataset, ABC):

  IS_CLASSIFICATION = False
  IS_ONLINE_VOXELIZATION = False
  NEED_PRED_POSTPROCESSING = False
  IS_FULL_POINTCLOUD_EVAL = False

  def __init__(self,
               data_paths,
               input_transform=None,
               target_transform=None,
               cache=False,
               data_root='/'):
    """
    data_paths: list of lists, [[str_path_to_input, str_path_to_label], [...]]
    """
    Dataset.__init__(self)

    # Allows easier path concatenation
    if not isinstance(data_root, Path):
      data_root = Path(data_root)

    self.data_root = data_root
    # self.data_paths = sorted(data_paths)
    self.data_paths = data_paths
    self.input_transform = input_transform
    self.target_transform = target_transform

    # dictionary of input
    self.data_loader_dict = {
        'input': (self.load_input, self.input_transform),
        'target': (self.load_target, self.target_transform)
    }

    # For large dataset, do not cache
    self.cache = cache
    self.cache_dict = defaultdict(dict)
    self.loading_key_order = ['input', 'target']

  def load_ply(self, index, data_index=0):
    # base class, not used
    filepath = self.data_root / self.data_paths[index][data_index]
    return self.read_ply(filepath)

  def load_input(self, index):
    raise NotImplementedError

  def load_target(self, index):
    raise NotImplementedError

  def get_classnames(self):
    pass

  def reorder_result(self, result):
    return result

  def __getitem__(self, index):
    out_array = []
    for k in self.loading_key_order:
      loader, transformer = self.data_loader_dict[k]
      v = loader(index)
      if transformer:
        v = transformer(v)
      out_array.append(v)
    return out_array

  def __len__(self):
    return len(self.data_paths)

class VoxelizationDatasetBase(DictDataset, ABC):
  IS_TEMPORAL = False
  CLIP_SIZE = 1000
  CLIP_BOUND = (-1000, -1000, -1000, 1000, 1000, 1000)
  ROTATION_AXIS = None
  LOCFEAT_IDX = None
  TRANSLATION_AUG = 0.
  INPUT_SPATIAL_DIM = (128, 128, 128)
  OUTPUT_SPATIAL_DIM = (128, 128, 128)
  NUM_IN_CHANNEL = None
  NUM_LABELS = -1  # Number of labels in the dataset, including all ignore classes
  IGNORE_LABELS = None  # labels that are not evaluated
  IS_ONLINE_VOXELIZATION = True

  def __init__(self,
               data_paths,
               input_transform=None,
               target_transform=None,
               cache=False,
               data_root='/',
               explicit_rotation=-1,
               ignore_mask=-1,
               return_transformation=False,
               **kwargs):
    """
    ignore_mask: label value for ignore class. It will not be used as a class in the loss or evaluation.
    explicit_rotation: # of discretization of 360 degree. # data would be num_data * explicit_rotation
    """
    DictDataset.__init__(
        self,
        data_paths,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        data_root=data_root)

    self.ignore_mask = ignore_mask
    self.explicit_rotation = explicit_rotation
    self.return_transformation = return_transformation

  def __getitem__(self, index):
    raise NotImplementedError

  def load_ply(self, index):
    if not self.load_whole:
        filepath = self.data_root / self.data_paths[index]
        return read_plyfile(filepath), None
    # readply get 7 chs: xyz-rgb-label
    else:
        feats = self.data_dict['data'][index]
        label = self.data_dict['label'][index]
        return np.concatenate([feats, label.reshape(-1,1)], axis=-1), None

  def __len__(self):
    num_data = len(self.data_paths)
    if self.explicit_rotation > 1:
      return num_data * self.explicit_rotation
    return num_data


class SparseVoxelizationDataset(VoxelizationDatasetBase):
  """This dataset loads RGB point clouds and their labels as a list of points
  and voxelizes the pointcloud with sufficient data augmentation.
  """
  # Voxelization arguments
  CLIP_BOUND = None
  VOXEL_SIZE = 0.05  # 5cm

  # Augmentation arguments
  SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 6, np.pi / 6), (-np.pi, np.pi), (-np.pi / 6, np.pi / 6))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.05, 0.05), (-0.2, 0.2))
  ELASTIC_DISTORT_PARAMS = None
  PREVOXELIZE_VOXEL_SIZE = None

  def __init__(self,
               data_paths,
               input_transform=None,
               target_transform=None,
               data_root='/',
               explicit_rotation=-1,
               # ignore_label=255,
               ignore_label= -1,
               return_transformation=False,
               augment_data=False,
               elastic_distortion=False,
               config=None,
               phase=DatasetPhase.Train,
               **kwargs):

    self.augment_data = augment_data
    self.elastic_distortion = elastic_distortion
    self.config = config
    self.phase = phase
    VoxelizationDatasetBase.__init__(
        self,
        data_paths,
        input_transform=input_transform,
        target_transform=target_transform,
        cache=cache,
        data_root=data_root,
        ignore_mask=ignore_label,
        return_transformation=return_transformation)

    self.sparse_voxelizer = SparseVoxelizer(
        voxel_size=config.voxel_size,
        clip_bound=self.CLIP_BOUND,
        use_augmentation=augment_data,
        scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
        rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
        translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
        rotation_axis=self.LOCFEAT_IDX,
        ignore_label=ignore_label)

    # loading the whole dataset from the pth file but not all ply files
    # if load-whole: the pre-defined pth already processed the label mapping, so should skip this part
    if self.phase == DatasetPhase.Train:
        self.split = 'train'
    elif self.phase == DatasetPhase.Val:
        self.split = 'val'
    else:
        raise NotImplementedError

    if hasattr(config, 'use_aux') and config.use_aux:
        # assert config.load_whole is True  # load_whole to avoid buggy, since the loaded file order are not consistent(e.g. scene 1 with different idx)

        aux_path = self.config.log_dir + 'preds_{}.pth'.format(self.split)
        assert os.path.exists(aux_path),  "No Aux file found, link the `preds_{}` file into the log_dir"
        self.aux_data = torch.load(aux_path, 'cpu')['pred']

    if hasattr(config, "load_whole") and config.load_whole:
        self.load_whole = True
        datapath = "/data/eva_share_users/zhaotianchen/scannet/raw/scannet_pickles/"

        datapath = datapath + 'new_{}.pth'.format(self.split)
        self.data_dict = torch.load(datapath, 'cpu')
    else:
        self.load_whole = False

    if not self.load_whole:
        # map labels not evaluated to ignore_label
        label_map = {}
        n_used = 0
        for l in range(self.NUM_LABELS):
          if l in self.IGNORE_LABELS:
            label_map[l] = self.ignore_mask
          else:
            label_map[l] = n_used
            n_used += 1
        label_map[self.ignore_mask] = self.ignore_mask
        self.label_map = label_map
        self.NUM_LABELS -= len(self.IGNORE_LABELS)
    else:
        # for load-whole, still need to change the num_labels to 20
        self.NUM_LABELS -= len(self.IGNORE_LABELS)

  def get_output_id(self, iteration):
    return self.data_paths[iteration]

  def convert_mat2cfl(self, mat):
    # Generally, xyz,rgb,label
    return mat[:, :3], mat[:, 3:-1], mat[:, -1]

  def _augment_elastic_distortion(self, pointcloud):
    if self.ELASTIC_DISTORT_PARAMS is not None:
      random.seed(123)
      if random.random() < 0.95:
        for granularity, magnitude in self.ELASTIC_DISTORT_PARAMS:
          pointcloud = t.elastic_distortion(pointcloud, granularity, magnitude)
    return pointcloud

  def __getitem__(self, index):

    if self.explicit_rotation > 1:
      rotation_space = np.linspace(-np.pi, np.pi, self.explicit_rotation + 1)
      rotation_angle = rotation_space[index % self.explicit_rotation]
      index //= self.explicit_rotation
    else:
      rotation_angle = None

    # pointcloud, center = self.load_ply(0) # DEBUG ONLY!
    pointcloud, center = self.load_ply(index)
    pointcloud = pointcloud.astype('float32') # when load_whole, data from pth file stays float64
    if self.PREVOXELIZE_VOXEL_SIZE is not None:
      inds = ME.SparseVoxelize(pointcloud[:, :3] / self.PREVOXELIZE_VOXEL_SIZE, return_index=True)
      pointcloud = pointcloud[inds]

    if self.elastic_distortion:
      pointcloud = self._augment_elastic_distortion(pointcloud)

    # import open3d as o3d
    # from lib.open3d_utils import make_pointcloud
    # pcd = make_pointcloud(np.floor(pointcloud[:, :3] / self.PREVOXELIZE_VOXEL_SIZE))
    # o3d.draw_geometries([pcd])

    coords, feats, labels = self.convert_mat2cfl(pointcloud)

    # d = {}
    # d['coords'] = coords
    # d['feats'] = feats
    # d['labels'] = labels
    # torch.save(d,'pc.pth')

    outs = self.sparse_voxelizer.voxelize(
        coords,
        feats,
        labels,
        center=center,
        rotation_angle=rotation_angle,
        return_transformation=self.return_transformation)

    if self.return_transformation:
      coords, feats, labels, unique_map, inverse_map, transformation = outs
      transformation = np.expand_dims(transformation, 0)
    else:
      coords, feats, labels, unique_map, inverse_map = outs

    if self.config.use_aux:
        aux = self.aux_data[index]
        # check if saved is tensor
        if isinstance(aux, torch.Tensor):
            aux = aux[unique_map].numpy()
        else:
            aux = aux[unique_map]
            pass
        aux = aux + 1 # align with preds

    if self.config.is_export:
      self.input_transform = None
      self.target_transform = None

    # d = {}
    # d['coords'] = coords
    # d['feats'] = feats
    # d['labels'] = labels
    # d['aux'] = aux
    # torch.save(d,'voxel.pth')

    if self.config.use_aux:
        # cat the aux into labels
        labels = np.stack([labels, aux], axis=1)

    # map labels not used for evaluation to ignore_label
    if self.input_transform is not None:
        coords, feats, labels = self.input_transform(coords, feats, labels)
    if self.target_transform is not None:
        coords, feats, labels = self.target_transform(coords, feats, labels)
    if self.IGNORE_LABELS is not None:
        if self.load_whole:
            labels = labels - 1  # should align all labels to [-1, 20]
        else:
            labels = np.array([self.label_map[x] for x in labels], dtype=np.int)

    # for load-whole, the aux are in [0-19] 20 classes
    # but labels have [-1, 19]
    # d = {}
    # d['coords'] = coords
    # d['feats'] = feats
    # d['labels'] = labels[:,0]
    # d['aux'] = labels[:,1]
    # torch.save(d,'pre.pth')

    return_args = [coords, feats, labels, unique_map, inverse_map]
    if self.return_transformation:
      return_args.extend([pointcloud.astype(np.float32), transformation.astype(np.float32)])
    return tuple(return_args)

  def cleanup(self):
    self.sparse_voxelizer.cleanup()

def initialize_data_loader(DatasetClass,
                           config,
                           phase,
                           threads,
                           shuffle,
                           repeat,
                           augment_data,
                           batch_size,
                           limit_numpoints,
                           elastic_distortion=False,
                           input_transform=None,
                           target_transform=None):
  if isinstance(phase, str):
    phase = str2datasetphase_type(phase)

  if config.return_transformation:
    collate_fn = t.cflt_collate_fn_factory(limit_numpoints)
  else:
    collate_fn = t.cfl_collate_fn_factory(limit_numpoints)

  input_transforms = []
  if input_transform is not None:
    input_transforms += input_transform

  if augment_data:
    input_transforms += [
        t.RandomDropout(0.2),
        t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
        t.ChromaticAutoContrast(),
        t.ChromaticTranslation(config.data_aug_color_trans_ratio),
        t.ChromaticJitter(config.data_aug_color_jitter_std),
        # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
    ]

  if len(input_transforms) > 0:
    input_transforms = t.Compose(input_transforms)
  else:
    input_transforms = None

  dataset = DatasetClass(
      config,
      input_transform=input_transforms,
      target_transform=target_transform,
      cache=config.cache_data,
      augment_data=augment_data,
      elastic_distortion=elastic_distortion,
      phase=phase)

  if repeat:
    # Use the inf random sampler
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=threads,
        batch_size=batch_size,
        collate_fn=collate_fn,
        worker_init_fn=_init_fn,
        pin_memory=True,
        sampler=InfSampler(dataset, shuffle))
  else:
    # Default shuffle=False
    data_loader = DataLoader(
        dataset=dataset,
        num_workers=threads,
        batch_size=batch_size,
        collate_fn=collate_fn,
        worker_init_fn=_init_fn,
        pin_memory=True,
        shuffle=shuffle)

  return data_loader

def _init_fn(worker_id):
        np.random.seed(2021)

def generate_meta(voxels_path,
                  split_path,
                  get_area_fn,
                  trainarea,
                  valarea,
                  testarea,
                  data_root='/',
                  check_pc=False):
  train_file_list = []
  val_file_list = []
  test_file_list = []
  for pointcloud_file in tqdm(glob.glob(osp.join(data_root, voxels_path))):
    area = get_area_fn(pointcloud_file)
    if area in trainarea:
      file_list = train_file_list
    elif area in valarea:
      file_list = val_file_list
    elif area in testarea:
      file_list = test_file_list
    else:
      raise ValueError('Area %s not in the split' % area)

    # Skip label files.
    if pointcloud_file.endswith('_label_voxel.ply'):
      continue

    # Parse and check if the corresponding label file exists.
    file_stem, file_ext = osp.splitext(pointcloud_file)
    file_stem_split = file_stem.split('_')
    file_stem_split.insert(-1, 'label')

    pointcloud_label_file = '_'.join(file_stem_split) + file_ext
    if not osp.isfile(pointcloud_label_file):
      raise ValueError('Lable file missing for: ' + pointcloud_file)

    # Check if the pointcloud is empty.
    if check_pc:
      pointcloud_data = read_plyfile(pointcloud_file)
      if not pointcloud_data:
        print('Skipping empty point cloud: %s.')
        continue

    pointcloud_file = osp.relpath(pointcloud_file, data_root)
    pointcloud_label_file = osp.relpath(pointcloud_label_file, data_root)

    # Append metadata.
    file_list.append([pointcloud_file, pointcloud_label_file])

  with open(split_path % 'train', 'w') as f:
    f.write('\n'.join([' '.join(pair) for pair in train_file_list]))
  with open(split_path % 'val', 'w') as f:
    f.write('\n'.join([' '.join(pair) for pair in val_file_list]))
  with open(split_path % 'test', 'w') as f:
    f.write('\n'.join([' '.join(pair) for pair in test_file_list]))
