# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import json
import random
import logging
import os
import errno
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def elementwise_multiplication(x, y, n):

    def is_iterable(z):
        if isinstance(z, (list, tuple)):
            return True
        else:
            assert type(z) is int
            return False

    if is_iterable(x) and is_iterable(y):
        assert len(x) == len(y) == n

    def convert_to_iterable(z):
        if is_iterable(z):
            return z
        else:
            return [
                    z,
            ] * n

    x = convert_to_iterable(x)
    y = convert_to_iterable(y)
    return [a * b for a, b in zip(x, y)]


def load_state_with_same_shape(model, weights):
    model_state = model.state_dict()
    filtered_weights = {
            k: v for k, v in weights.items() if k in model_state and v.size() == model_state[k].size()
    }
    logging.info("Loading weights:" + ', '.join(filtered_weights.keys()))
    return filtered_weights


def checkpoint(model, optimizer, epoch, iteration, config, best_val=None, best_val_iter=None, postfix=None, save_inter=False):
    mkdir_p(config.log_dir)
    if config.overwrite_weights:
        if postfix is not None:
            filename = f"checkpoint_{config.model}{postfix}.pth"
        else:
            filename = f"checkpoint_{config.model}.pth"
    else:
        filename = f"checkpoint_{config.model}_iter_{iteration}.pth"
    checkpoint_file = config.log_dir + '/' + filename
    state = {
            'iteration': iteration,
            'epoch': epoch,
            'arch': config.model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
    }
    if best_val is not None:
        state['best_val'] = best_val
        state['best_val_iter'] = best_val_iter
    json.dump(vars(config), open(config.log_dir + '/config.json', 'w'), indent=4)
    torch.save(state, checkpoint_file)
    logging.info(f"Checkpoint saved to {checkpoint_file}")
    # Delete symlink if it exists
    if os.path.exists(f'{config.log_dir}weights.pth'):
        os.remove(f'{config.log_dir}weights.pth')
    # Create symlink
    os.system(f'cd {config.log_dir}; ln -s {filename} weights.pth')

    # save intermediate
    if save_inter:
        inter_d = {}
        # due to some stupid bug, some xyz_map in the submodule are not in model.named_modules(), dirty fix here
        for n_block,m_block in model.named_children():
            for n,m in m_block.named_buffers():
                if 'map' in n:
                    inter_d[n_block + '.' + n] = m
        # for n,m in model.named_buffers():
            # if 'map' in n:
                # inter_d[n] = m
        torch.save(inter_d, os.path.join(config.log_dir, 'map.pth'))
        logging.info('Map saved in {}'.format(config.log_dir))

def save_map(model, config):
    inter_d = {}
    for n,m in model.named_buffers():
        if 'map' in n:
            inter_d[n] = m
    torch.save(inter_d, os.path.join(config.log_dir, 'val_map.pth'))
    logging.info('Map saved in {}'.format(config.log_dir))

def feat_augmentation(data, normalized, config):
    # color shift
    if random.random() < 0.9:
        tr = (torch.rand(1, 3).type_as(data) - 0.5) * \
                config.data_aug_max_color_trans
        if normalized:
            tr /= 255
        data[:, :3] += tr

    # color jitter
    if random.random() < 0.9:
        noise = torch.randn((data.size(0), 3), dtype=data.dtype).type_as(data)
        noise *= config.data_aug_noise_std if normalized else 255 * config.data_aug_noise_std
        data[:, :3] += noise

    # height jitter
    if data.size(1) > 3 and random.random() < 0.9:
        data[:, -1] += torch.randn(1, dtype=data.dtype).type_as(data)

    if data.size(1) > 3 and random.random() < 0.9:
        data[:, -1] += torch.randn((data.size(0)), dtype=data.dtype).type_as(data)

    # normal jitter
    if data.size(1) > 6 and random.random() < 0.9:
        data[:, 3:6] += torch.randn((data.size(0), 3), dtype=data.dtype).type_as(data)


def precision_at_one(pred, target, ignore_label=-1):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != ignore_label]
    correct = correct.view(-1)
    if correct.nelement():
        return correct.float().sum(0).mul(100.0 / correct.size(0)).item()
    else:
        return float('nan')


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


class WithTimer(object):
    """Timer for with statement."""

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        out_str = 'Elapsed: %s' % (time.time() - self.tstart)
        if self.name:
            logging.info('[{self.name}]')
        logging.info(out_str)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.averate_time = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class ExpTimer(Timer):
    """ Exponential Moving Average Timer """

    def __init__(self, alpha=0.5):
        super(ExpTimer, self).__init__()
        self.alpha = alpha

    def toc(self):
        self.diff = time.time() - self.start_time
        self.average_time = self.alpha * self.diff + \
                (1 - self.alpha) * self.average_time
        return self.average_time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


def debug_on():
    import sys
    import pdb
    import functools
    import traceback

    def decorator(f):

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception:
                info = sys.exc_info()
                traceback.print_exception(*info)
                pdb.post_mortem(info[2])

        return wrapper

    return decorator


def permute_label(model, soutput, target, num_labels, ignore_label=-1):
    if model.NETWORK_TYPE.name == 'CLASSIFICATION':
        perm = model.get_coords(0)[:, -1]
        return target[perm.long()]
    else:
        assert (target >= num_labels).sum() == (target == ignore_label).sum()
        clipped_target = target.clone()
        clipped_target[target == ignore_label] = num_labels
        permuted_target = soutput.C.permute_label(
                clipped_target, num_labels + 1, target_pixel_dist=model.OUT_PIXEL_DIST, label_pixel_dist=1)
        permuted_target[permuted_target == num_labels] = ignore_label
        return permuted_target.int()


def get_prediction(dataset, output, target):
    if dataset.NEED_PRED_POSTPROCESSING:
        return dataset.get_prediction(output, target)
    else:
        return output.max(1)[1]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_torch_device(is_cuda):
    return torch.device('cuda' if is_cuda else 'cpu')

# gumbel softmax
def sample_gumbel(shape, device, eps=1e-20):
    uniform_rand = torch.rand(shape).to(device)
    return Variable(-torch.log(-torch.log(uniform_rand + eps) + eps))

def gumbel_softmax_sample(logits, temperature, eps=None):
    if eps is None:
        eps = sample_gumbel(logits.size(), logits.device)
    y = logits + eps
    return F.softmax(y / temperature, dim=-1), eps

def gumbel_softmax(logits, temperature, eps=None, hard=True):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y, eps = gumbel_softmax_sample(logits, temperature, eps)
    if hard:
        y = straight_through(y)
    return y, eps

# bernoulli sample
def relaxed_bernoulli_sample(logits, temperature):
    relaxed_bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature, logits=logits)
    y = relaxed_bernoulli.rsample()
    return y

def straight_through(y):
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def mask2d(batch_size, dim, keep_prob, device):
    mask = torch.floor(torch.rand(batch_size, dim, device=device) + keep_prob) / keep_prob
    return mask


