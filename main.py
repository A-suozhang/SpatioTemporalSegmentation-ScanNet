# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
# Change dataloader multiprocess start method to anything not fork
import torch.multiprocessing as mp
try:
    mp.set_start_method('forkserver')  # Reuse process created
except RuntimeError:
    pass

import os
import sys
import json
import logging
from easydict import EasyDict as edict

import random
import numpy as np

# Torch packages
import torch

# Train deps
from config import get_config
import shutil

from lib.test import test
from lib.train import train
from lib.utils import load_state_with_same_shape, get_torch_device, count_parameters
from lib.dataset import initialize_data_loader
from lib.datasets import load_dataset

from models import load_model

import MinkowskiEngine as ME    # force loadding

def setup_seed(seed):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)
                torch.backends.cudnn.deterministic = True


def main():
    config = get_config()

    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(config.log_dir, './model.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logging.basicConfig(
                format=os.uname()[1].split('.')[0] + ' %(asctime)s %(message)s',
                datefmt='%m/%d %H:%M:%S',
                handlers=[ch, file_handler])

    if config.test_config:
        json_config = json.load(open(config.test_config, 'r'))
        json_config['is_train'] = False
        json_config['weights'] = config.weights
        config = edict(json_config)
    elif config.resume:
        json_config = json.load(open(config.resume + '/config.json', 'r'))
        json_config['resume'] = config.resume
        config = edict(json_config)
    else:
        '''bakup files'''
        if not os.path.exists(os.path.join(config.log_dir,'models')):
                os.mkdir(os.path.join(config.log_dir,'models'))
                for filename in os.listdir('./models'):
                        if ".py" in filename:
                                shutil.copy(os.path.join("./models", filename), os.path.join(config.log_dir,'models'))
                shutil.copy('./main.py', config.log_dir)


    if config.is_cuda and not torch.cuda.is_available():
        raise Exception("No GPU found")
    device = get_torch_device(config.is_cuda)

    # torch.set_num_threads(config.threads)
    # torch.manual_seed(config.seed)
    # if config.is_cuda:
    #       torch.cuda.manual_seed(config.seed)

    logging.info('===> Configurations')
    dconfig = vars(config)
    for k in dconfig:
        logging.info('      {}: {}'.format(k, dconfig[k]))

    DatasetClass = load_dataset(config.dataset)
    logging.info('===> Initializing dataloader')

    if config.is_train:
        setup_seed(2021)

        train_data_loader = initialize_data_loader(
                DatasetClass,
                config,
                phase=config.train_phase,
                threads=config.threads,
                # threads=8,
                augment_data=True,
                elastic_distortion=config.train_elastic_distortion,
                # elastic_distortion=False,
                # shuffle=True,
                shuffle=True,
                # repeat=True,
                repeat=True,
                batch_size=config.batch_size,
                # batch_size=8,
                limit_numpoints=config.train_limit_numpoints)

        # dat = iter(train_data_loader).__next__()
        # import ipdb; ipdb.set_trace()

        val_data_loader = initialize_data_loader(
                DatasetClass,
                config,
                # threads=0,
                threads=config.val_threads,
                phase=config.val_phase,
                augment_data=False,
                elastic_distortion=config.test_elastic_distortion,
                shuffle=False,
                repeat=False,
                # batch_size=config.val_batch_size,
                batch_size=8,
                limit_numpoints=False)

        # dat = iter(val_data_loader).__next__()
        # import ipdb; ipdb.set_trace()

        if train_data_loader.dataset.NUM_IN_CHANNEL is not None:
            num_in_channel = train_data_loader.dataset.NUM_IN_CHANNEL
        else:
            num_in_channel = 3

        num_labels = train_data_loader.dataset.NUM_LABELS
    else:
        test_data_loader = initialize_data_loader(
                DatasetClass,
                config,
                threads=config.threads,
                phase=config.test_phase,
                augment_data=False,
                elastic_distortion=config.test_elastic_distortion,
                shuffle=False,
                repeat=False,
                batch_size=config.test_batch_size,
                limit_numpoints=False)
        if test_data_loader.dataset.NUM_IN_CHANNEL is not None:
            num_in_channel = test_data_loader.dataset.NUM_IN_CHANNEL
        else:
            num_in_channel = 3

        num_labels = test_data_loader.dataset.NUM_LABELS

    logging.info('===> Building model')

    NetClass = load_model(config.model)
    model = NetClass(num_in_channel, num_labels, config)
    logging.info('===> Number of trainable parameters: {}: {}'.format(NetClass.__name__,count_parameters(model)))
    logging.info(model)

    # Set the number of threads
    # ME.initialize_nthreads(12, D=3)

    model = model.to(device)

    if config.weights == 'modelzoo':    # Load modelzoo weights if possible.
        logging.info('===> Loading modelzoo weights')
        model.preload_modelzoo()
    # Load weights if specified by the parameter.
    elif config.weights.lower() != 'none':
        logging.info('===> Loading weights: ' + config.weights)
        state = torch.load(config.weights)
        if config.weights_for_inner_model:
            model.model.load_state_dict(state['state_dict'])
        else:
            if config.lenient_weight_loading:
                matched_weights = load_state_with_same_shape(model, state['state_dict'])
                model_dict = model.state_dict()
                model_dict.update(matched_weights)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(state['state_dict'])

    if config.is_train:
        train(model, train_data_loader, val_data_loader, config)
    else:
        test(model, test_data_loader, config)


if __name__ == '__main__':
    __spec__ = None
    main()
