# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import logging
import os.path as osp

import torch
from torch import nn

from tensorboardX import SummaryWriter

from lib.test import test, test_points
from lib.utils import checkpoint, precision_at_one, \
        Timer, AverageMeter, get_prediction, get_torch_device
from lib.solvers import initialize_optimizer, initialize_scheduler

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

import numpy as np


def validate(model, val_data_loader, writer, curr_iter, config, transform_data_fn=None):
    v_loss, v_score, v_mAP, v_mIoU = test(model, val_data_loader, config)
    writer.add_scalar('validation/mIoU', v_mIoU, curr_iter)
    writer.add_scalar('validation/loss', v_loss, curr_iter)
    writer.add_scalar('validation/precision_at_1', v_score, curr_iter)
    return v_mIoU

def train(model, data_loader, val_data_loader, config, transform_data_fn=None):

    device = get_torch_device(config.is_cuda)
    # Set up the train flag for batch normalization
    model.train()

    # Configuration
    writer = SummaryWriter(log_dir=config.log_dir)
    data_timer, iter_timer = Timer(), Timer()
    data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
    losses, scores = AverageMeter(), AverageMeter()

    optimizer = initialize_optimizer(model.parameters(), config)
    scheduler = initialize_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)

    writer = SummaryWriter(log_dir=config.log_dir)

    # Train the network
    logging.info('===> Start training')
    best_val_miou, best_val_iter, curr_iter, epoch, is_training = 0, 0, 1, 1, True

    if config.resume:
        checkpoint_fn = config.resume + '/weights.pth'
        if osp.isfile(checkpoint_fn):
            logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn)
            curr_iter = state['iteration'] + 1
            epoch = state['epoch']
            d = {k:v for k,v in state['state_dict'].items() if 'map' not in k }
            model.load_state_dict(d)
            if config.resume_optimizer:
                scheduler = initialize_scheduler(optimizer, config, last_step=curr_iter)
                optimizer.load_state_dict(state['optimizer'])
            if 'best_val' in state:
                best_val_miou = state['best_val']
                best_val_iter = state['best_val_iter']
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

    data_iter = data_loader.__iter__()
    while is_training:
        for iteration in range(len(data_loader) // config.iter_size):
            optimizer.zero_grad()
            data_time, batch_loss = 0, 0
            iter_timer.tic()

            for sub_iter in range(config.iter_size):
                # Get training data
                data_timer.tic()
                if config.return_transformation:
                    coords, input, target, pointcloud, transformation = data_iter.next()
                else:
                    coords, input, target = data_iter.next()

                # For some networks, making the network invariant to even, odd coords is important
                coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

                # Preprocess input
                if config.normalize_color:
                    input[:, :3] = input[:, :3] / 255. - 0.5

                sinput = SparseTensor(input, coords, device=device)
                data_time += data_timer.toc(False)
                # model.initialize_coords(*init_args)
                soutput = model(sinput)
                # The output of the network is not sorted
                target = target.view(-1).long().to(device)

                loss = criterion(soutput.F, target.long())

                # Compute and accumulate gradient
                loss /= config.iter_size
                batch_loss += loss.item()
                loss.backward()

            # Update number of steps
            optimizer.step()
            scheduler.step()

            # CLEAR CACHE!
            torch.cuda.empty_cache()

            data_time_avg.update(data_time)
            iter_time_avg.update(iter_timer.toc(False))

            pred = get_prediction(data_loader.dataset, soutput.F, target)
            score = precision_at_one(pred, target)
            losses.update(batch_loss, target.size(0))
            scores.update(score, target.size(0))

            if curr_iter >= config.max_iter:
                is_training = False
                break

            if curr_iter % config.stat_freq == 0 or curr_iter == 1:
                lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
                debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
                        epoch, curr_iter,
                        len(data_loader) // config.iter_size, losses.avg, lrs)
                debug_str += "Score {:.3f}\tData time: {:.4f}, Iter time: {:.4f}".format(
                        scores.avg, data_time_avg.avg, iter_time_avg.avg)
                logging.info(debug_str)
                # Reset timers
                data_time_avg.reset()
                iter_time_avg.reset()
                # Write logs
                writer.add_scalar('training/loss', losses.avg, curr_iter)
                writer.add_scalar('training/precision_at_1', scores.avg, curr_iter)
                writer.add_scalar('training/learning_rate', scheduler.get_lr()[0], curr_iter)
                losses.reset()
                scores.reset()

            # Save current status, save before val to prevent occational mem overflow
            if curr_iter % config.save_freq == 0:
                checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, save_inter=True)


            # Validation
            if curr_iter % config.val_freq == 0:
                val_miou = validate(model, val_data_loader, writer, curr_iter, config, transform_data_fn)
                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    best_val_iter = curr_iter
                    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter,
                                         "best_val", save_inter=True)
                logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

                # Recover back
                model.train()

            # End of iteration
            curr_iter += 1

        epoch += 1

    # Explicit memory cleanup
    if hasattr(data_iter, 'cleanup'):
        data_iter.cleanup()

    # Save the final model
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)
    v_loss, v_score, v_mAP, val_mIoU = test(model, val_data_loader, config)
    if val_miou > best_val_miou:
        best_val_miou = val_miou
        best_val_iter = curr_iter
        checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, "best_val")
    logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))


def train_point(model, data_loader, val_data_loader, config, transform_data_fn=None):

    device = get_torch_device(config.is_cuda)
    # Set up the train flag for batch normalization
    model.train()

    # Configuration
    writer = SummaryWriter(log_dir=config.log_dir)
    data_timer, iter_timer = Timer(), Timer()
    data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
    losses, scores = AverageMeter(), AverageMeter()

    optimizer = initialize_optimizer(model.parameters(), config)
    scheduler = initialize_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    writer = SummaryWriter(log_dir=config.log_dir)

    # Train the network
    logging.info('===> Start training')
    best_val_miou, best_val_iter, curr_iter, epoch, is_training = 0, 0, 1, 1, True

    if config.resume:
        checkpoint_fn = config.resume + '/weights.pth'
        if osp.isfile(checkpoint_fn):
            logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn)
            curr_iter = state['iteration'] + 1
            epoch = state['epoch']
            d = {k:v for k,v in state['state_dict'].items() if 'map' not in k }
            model.load_state_dict(d)
            if config.resume_optimizer:
                scheduler = initialize_scheduler(optimizer, config, last_step=curr_iter)
                optimizer.load_state_dict(state['optimizer'])
            if 'best_val' in state:
                best_val_miou = state['best_val']
                best_val_iter = state['best_val_iter']
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

    data_iter = data_loader.__iter__()
    while is_training:

        num_class = 20
        total_correct_class = torch.zeros(num_class, device=device)
        total_iou_deno_class = torch.zeros(num_class, device=device)

        for iteration in range(len(data_loader) // config.iter_size):
            optimizer.zero_grad()
            data_time, batch_loss = 0, 0
            iter_timer.tic()
            for sub_iter in range(config.iter_size):
                # Get training data
                data= data_iter.next()
                points, target, sample_weight = data
                if config.pure_point:

                        sinput = points.transpose(1,2).cuda().float()

                        # DEBUG: use the discrete coord for point-based

                        feats = torch.unbind(points[:,:,:], dim=0)
                        voxel_size = config.voxel_size
                        coords = torch.unbind(points[:,:,:3]/voxel_size, dim=0)  # 0.05 is the voxel-size
                        coords, feats= ME.utils.sparse_collate(coords, feats)
                        # assert feats.reshape([16, 4096, -1]) == points[:,:,3:]
                        points_ = ME.TensorField(features=feats.float(), coordinates=coords, device=device)
                        tmp_voxel = points_.sparse()
                        sinput_ = tmp_voxel.slice(points_)
                        sinput = torch.cat([sinput_.C[:,1:]*config.voxel_size, sinput_.F[:,3:]],dim=1).reshape([config.batch_size, config.num_points, 6])
                        # sinput = sinput_.F.reshape([config.batch_size, config.num_points, 6])
                        sinput = sinput.transpose(1,2).cuda().float()

                        # sinput = torch.cat([coords[:,1:], feats],dim=1).reshape([config.batch_size, config.num_points, 6])
                        # sinput = sinput.transpose(1,2).cuda().float()


                        # For some networks, making the network invariant to even, odd coords is important
                        # coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

                        # Preprocess input
                        # if config.normalize_color:
                            # feats = feats / 255. - 0.5

                        # torch.save(points[:,:,:3], './sandbox/tensorfield-c.pth')
                        # torch.save(points_.C, './sandbox/points-c.pth')


                else:
                        # feats = torch.unbind(points[:,:,3:], dim=0) # WRONG: should also feed in xyz as inupt feature
                        voxel_size = config.voxel_size
                        coords = torch.unbind(points[:,:,:3]/voxel_size, dim=0)  # 0.05 is the voxel-size
                        # Normalize the xyz in feature
                        # points[:,:,:3] = points[:,:,:3] / points[:,:,:3].mean()
                        feats = torch.unbind(points[:,:,:], dim=0)
                        coords, feats= ME.utils.sparse_collate(coords, feats)

                        # For some networks, making the network invariant to even, odd coords is important
                        # coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

                        # Preprocess input
                        # if config.normalize_color:
                            # feats = feats / 255. - 0.5

                        # they are the same
                        points_ = ME.TensorField(features=feats.float(), coordinates=coords, device=device)
                        # points_1 = ME.TensorField(features=feats.float(), coordinates=coords, device=device, quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
                        # points_2 = ME.TensorField(features=feats.float(), coordinates=coords, device=device, quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
                        sinput = points_.sparse()

                data_time += data_timer.toc(False)
                B, npoint = target.shape

                # model.initialize_coords(*init_args)
                soutput = model(sinput)
                if config.pure_point:
                        soutput = soutput.reshape([B*npoint, -1])
                else:
                        soutput = soutput.slice(points_).F
                        # s1 = soutput.slice(points_)
                        # print(soutput.quantization_mode)
                        # soutput.quantization_mode = ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE
                        # s2 = soutput.slice(points_)

                # The output of the network is not sorted
                target = (target-1).view(-1).long().to(device)

                # catch NAN
                if torch.isnan(soutput).sum() > 0:
                        import ipdb; ipdb.set_trace()

                loss = criterion(soutput, target)

                if torch.isnan(loss).sum()>0:
                        import ipdb; ipdb.set_trace()

                loss = (loss*sample_weight.to(device)).mean()

                # Compute and accumulate gradient
                loss /= config.iter_size
                batch_loss += loss.item()
                loss.backward()

            # Update number of steps
            optimizer.step()
            scheduler.step()

            # CLEAR CACHE!
            torch.cuda.empty_cache()


            data_time_avg.update(data_time)
            iter_time_avg.update(iter_timer.toc(False))

            pred = get_prediction(data_loader.dataset, soutput, target)
            score = precision_at_one(pred, target, ignore_label=-1)
            losses.update(batch_loss, target.size(0))
            scores.update(score, target.size(0))

            # Calc the iou
            for l in range(num_class):
                total_correct_class[l] += ((pred == l) & (target == l)).sum()
                total_iou_deno_class[l] += (((pred == l) & (target >= 0)) | (target == l)).sum()

            if curr_iter >= config.max_iter:
                is_training = False
                break

            if curr_iter % config.stat_freq == 0 or curr_iter == 1:
                lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
                debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
                        epoch, curr_iter,
                        len(data_loader) // config.iter_size, losses.avg, lrs)
                debug_str += "Score {:.3f}\tData time: {:.4f}, Iter time: {:.4f}".format(
                        scores.avg, data_time_avg.avg, iter_time_avg.avg)
                logging.info(debug_str)
                # Reset timers
                data_time_avg.reset()
                iter_time_avg.reset()
                # Write logs
                writer.add_scalar('training/loss', losses.avg, curr_iter)
                writer.add_scalar('training/precision_at_1', scores.avg, curr_iter)
                writer.add_scalar('training/learning_rate', scheduler.get_lr()[0], curr_iter)
                losses.reset()
                scores.reset()

            # Save current status, save before val to prevent occational mem overflow
            if curr_iter % config.save_freq == 0:
                checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, save_inter=True)

            # Validation
            # if curr_iter % config.val_freq == 0:
                # val_miou = test_points(model, val_data_loader, writer, curr_iter, config, transform_data_fn)
                # if val_miou > best_val_miou:
                    # best_val_miou = val_miou
                    # best_val_iter = curr_iter
                    # checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter,
                                         # "best_val")
                # logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

                # # Recover back
                # model.train()

            # End of iteration
            curr_iter += 1

        IoU = (total_correct_class) / (total_iou_deno_class+1e-6)
        logging.info('eval point avg class IoU: %f' % ((IoU).mean()*100.))

        epoch += 1


    # Explicit memory cleanup
    if hasattr(data_iter, 'cleanup'):
        data_iter.cleanup()

    # Save the final model
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)

    test_points(model, val_data_loader, config)
    # if val_miou > best_val_miou:
        # best_val_miou = val_miou
        # best_val_iter = curr_iter
        # checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, "best_val")
    # logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

