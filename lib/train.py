# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import logging
import os.path as osp

import torch
from torch import nn

from lib.test import test
from lib.utils import checkpoint, precision_at_one, \
        Timer, AverageMeter, get_prediction, get_torch_device
from lib.solvers import initialize_optimizer, initialize_scheduler

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

import numpy as np

from models import load_model
from models.pct_voxel_utils import separate_batch, voxel2points

# Profiler
from lib.profile import CUDAMemoryProfiler
from lib.sam import SAM
import sys
import threading

def validate(model, val_data_loader, writer, curr_iter, config, transform_data_fn=None):
    v_loss, v_score, v_mAP, v_mIoU = test(model, val_data_loader, config)
    return v_mIoU

def train(model, data_loader, val_data_loader, config, transform_data_fn=None):

    device = get_torch_device(config.is_cuda)
    # Set up the train flag for batch normalization
    model.train()

    # Configuration
    data_timer, iter_timer = Timer(), Timer()
    data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
    regs, losses, scores = AverageMeter(), AverageMeter(), AverageMeter()

    optimizer = initialize_optimizer(model.parameters(), config)
    scheduler = initialize_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)

    # Train the network
    logging.info('===> Start training')
    best_val_miou, best_val_iter, curr_iter, epoch, is_training = 0, 0, 1, 1, True

    if config.resume:
        # Test loaded ckpt first
        v_loss, v_score, v_mAP, v_mIoU = test(model, val_data_loader, config)

        checkpoint_fn = config.resume + '/weights.pth'
        if osp.isfile(checkpoint_fn):
            logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn)
            curr_iter = state['iteration'] + 1
            epoch = state['epoch']
            # we skip attention maps because the shape won't match because voxel number is different
            # e.g. copyting a param with shape (23385, 8, 4) to (43529, 8, 4)
            d = {k:v for k,v in state['state_dict'].items() if 'map' not in k }
            # handle those attn maps we don't load from saved dict
            for k in model.state_dict().keys():
                if k in d.keys(): continue
                d[k] = model.state_dict()[k]
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
    if config.dataset == "SemanticKITTI":
        num_class = 19
        config.normalize_color = False
        config.xyz_input = False
        val_freq_ = config.val_freq
        config.val_freq = config.val_freq*10
    elif config.dataset == "S3DIS":
        num_class = 13
        config.normalize_color = False
        config.xyz_input = False
        val_freq_ = config.val_freq
        config.val_freq = config.val_freq
    elif config.dataset == "Nuscenes":
        num_class = 16
        config.normalize_color = False
        config.xyz_input = False
        val_freq_ = config.val_freq
        config.val_freq = config.val_freq*50
    else:
        num_class = 20
        val_freq_ = config.val_freq

    while is_training:
        total_correct_class = torch.zeros(num_class, device=device)
        total_iou_deno_class = torch.zeros(num_class, device=device)

        for iteration in range(len(data_loader) // config.iter_size):
            optimizer.zero_grad()
            data_time, batch_loss = 0, 0
            iter_timer.tic()

            if curr_iter >= config.max_iter:
                # if curr_iter >= max(config.max_iter, config.epochs*(len(data_loader) // config.iter_size):
                    is_training = False
                    break
            elif curr_iter >= config.max_iter*(2/3):
                config.val_freq = val_freq_*2 # valid more freq on lower half

            for sub_iter in range(config.iter_size):
                # Get training data
                data_timer.tic()
                pointcloud = None

                if config.return_transformation:
                    coords, input, target, _, _, pointcloud, transformation, _ = data_iter.next()
                else:
                    coords, input, target, _, _, _ = data_iter.next()  # ignore unique_map and inverse_map

                if config.use_aux:
                    assert target.shape[1] == 2
                    aux = target[:,1]
                    target = target[:,0]
                else:
                    aux = None

                # For some networks, making the network invariant to even, odd coords is important
                coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

                # Preprocess input
                if config.normalize_color:
                    input[:, :3] = input[:, :3] / input[:,:3].max() - 0.5
                    coords_norm = coords[:,1:] / coords[:,1:].max() - 0.5

                # cat xyz into the rgb feature
                if config.xyz_input:
                    input = torch.cat([coords_norm, input], dim=1)
                sinput = SparseTensor(input, coords, device=device)
                starget = SparseTensor(target.unsqueeze(-1).float(), coordinate_map_key=sinput.coordinate_map_key, coordinate_manager=sinput.coordinate_manager, device=device) # must share the same coord-manager to align for sinput

                data_time += data_timer.toc(False)
                # model.initialize_coords(*init_args)

                # d = {}
                # d['c'] = sinput.C
                # d['l'] = starget.F
                # torch.save('./plot/test-label.pth')
                # import ipdb; ipdb.set_trace()

                # Set up profiler
                # memory_profiler = CUDAMemoryProfiler(
                    # [model, criterion],
                    # filename="cuda_memory.profile"
                # )
                # sys.settrace(memory_profiler)
                # threading.settrace(memory_profiler)

                # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=True) as prof0:
                if aux is not None:
                    soutput = model(sinput, aux)
                elif config.enable_point_branch:
                    soutput = model(sinput, iter_ = curr_iter / config.max_iter, enable_point_branch=True)
                else:
                    # label-aux, feed it in as additional reg
                    soutput = model(sinput, iter_= curr_iter / config.max_iter, aux=starget)  # feed in the progress of training for annealing inside the model

                # The output of the network is not sorted
                target = target.view(-1).long().to(device)
                loss = criterion(soutput.F, target.long())

                # ====== other loss regs =====
                if hasattr(model, 'block1'):
                    cur_loss = torch.tensor([0.], device=device)

                    if hasattr(model.block1[0],'vq_loss'):
                        if model.block1[0].vq_loss is not None:
                            cur_loss = torch.tensor([0.], device=device)
                            for n, m in model.named_children():
                                if 'block' in n:
                                    cur_loss += m[0].vq_loss # m is the nn.Sequential obj, m[0] is the TRBlock
                            logging.info('Cur Loss: {}, Cur vq_loss: {}'.format(loss, cur_loss))
                            loss += cur_loss

                    if hasattr(model.block1[0],'diverse_loss'):
                        if model.block1[0].diverse_loss is not None:
                            cur_loss = torch.tensor([0.], device=device)
                            for n, m in model.named_children():
                                if 'block' in n:
                                    cur_loss += m[0].diverse_loss # m is the nn.Sequential obj, m[0] is the TRBlock
                            logging.info('Cur Loss: {}, Cur diverse _loss: {}'.format(loss, cur_loss))
                            loss += cur_loss

                    if hasattr(model.block1[0],'label_reg'):
                        if model.block1[0].label_reg is not None:
                            cur_loss = torch.tensor([0.], device=device)
                            for n, m in model.named_children():
                                if 'block' in n:
                                    cur_loss += m[0].label_reg # m is the nn.Sequential obj, m[0] is the TRBlock
                            # logging.info('Cur Loss: {}, Cur diverse _loss: {}'.format(loss, cur_loss))
                            loss += cur_loss

                # Compute and accumulate gradient
                loss /= config.iter_size
                batch_loss += loss.item()
                loss.backward()

                    # soutput = model(sinput)

            # Update number of steps
            if not config.use_sam:
                optimizer.step()
            else:
                optimizer.first_step(zero_grad=True)
                soutput = model(sinput, iter_= curr_iter / config.max_iter, aux=starget)
                criterion(soutput.F, target.long()).backward()
                optimizer.second_step(zero_grad=True)

            if config.lr_warmup is None:
                scheduler.step()
            else:
                if curr_iter >= config.lr_warmup:
                    scheduler.step()
                for g in optimizer.param_groups:
                    g['lr'] = config.lr*(iteration+1)/config.lr_warmup

            # CLEAR CACHE!
            torch.cuda.empty_cache()

            data_time_avg.update(data_time)
            iter_time_avg.update(iter_timer.toc(False))

            pred = get_prediction(data_loader.dataset, soutput.F, target)
            score = precision_at_one(pred, target, ignore_label=-1)

            regs.update(cur_loss.item(), target.size(0))
            losses.update(batch_loss, target.size(0))
            scores.update(score, target.size(0))

            # calc the train-iou
            for l in range(num_class):
                total_correct_class[l] += ((pred == l) & (target == l)).sum()
                total_iou_deno_class[l] += (((pred == l) & (target!=-1)) | (target == l) ).sum()

            if curr_iter % config.stat_freq == 0 or curr_iter == 1:
                lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
                IoU = ((total_correct_class) / (total_iou_deno_class+1e-6)).mean()*100.
                debug_str = "[{}] ===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
                        config.log_dir.split('/')[-2],
                        epoch, curr_iter,
                        len(data_loader) // config.iter_size, losses.avg, lrs)
                debug_str += "Score {:.3f}\tIoU {:.3f}\tData time: {:.4f}, Iter time: {:.4f}".format(
                        scores.avg, IoU.item(), data_time_avg.avg, iter_time_avg.avg)
                if regs.avg > 0:
                    debug_str += "\n Additional Reg Loss {:.3f}".format(regs.avg)
                # print(debug_str)
                logging.info(debug_str)
                # Reset timers
                data_time_avg.reset()
                iter_time_avg.reset()
                # Write logs
                losses.reset()
                scores.reset()

            # Save current status, save before val to prevent occational mem overflow
            if curr_iter % config.save_freq == 0:
                checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, save_inter=True)

            # Validation
            if curr_iter % config.val_freq == 0:
                val_miou = validate(model, val_data_loader, None, curr_iter, config, transform_data_fn)
                if val_miou > best_val_miou:
                    best_val_miou = val_miou
                    best_val_iter = curr_iter
                    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter,
                                         "best_val", save_inter=True)
                logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))
                # print("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

                # Recover back
                model.train()

            # End of iteration
            curr_iter += 1

        IoU = (total_correct_class) / (total_iou_deno_class+1e-6)
        logging.info('train point avg class IoU: %f' % ((IoU).mean()*100.))

        epoch += 1

    # Explicit memory cleanup
    if hasattr(data_iter, 'cleanup'):
        data_iter.cleanup()

    # Save the final model
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)
    v_loss, v_score, v_mAP, val_miou = test(model, val_data_loader, config)
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
    data_timer, iter_timer = Timer(), Timer()
    data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
    losses, scores = AverageMeter(), AverageMeter()

    optimizer = initialize_optimizer(model.parameters(), config)
    scheduler = initialize_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

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
                        '''

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
                        '''


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
                        coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

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
                # print(model.input_mlp[0].weight.max())
                # print(model.input_mlp[0].weight.grad.max())

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
                losses.reset()
                scores.reset()

            # Save current status, save before val to prevent occational mem overflow
            if curr_iter % config.save_freq == 0:
                checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, save_inter=True)

            # Validation:
            # for point-based should use alternate dataloader for eval
            # if curr_iter % config.val_freq == 0:
                # val_miou = test_points(model, val_data_loader, None, curr_iter, config, transform_data_fn)
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
        logging.info('train point avg class IoU: %f' % ((IoU).mean()*100.))

        epoch += 1


    # Explicit memory cleanup
    if hasattr(data_iter, 'cleanup'):
        data_iter.cleanup()

    # Save the final model
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)

    test_points(model, val_data_loader, config)
    if val_miou > best_val_miou:
        best_val_miou = val_miou
        best_val_iter = curr_iter
        checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, "best_val")
    logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))


def DistillLoss(tch_xs, stu_xs):

    # l2_loss = torch.zeros([B]).to(tch_xs[0].device)
    # print('start computing')
    l2_losses = []
    for tch_x,stu_x in zip(tch_xs, stu_xs):

        tch_xc, tch_xf, tch_x_idx = voxel2points(tch_x)
        stu_xc, stu_xf, stu_x_idx = voxel2points(stu_x)

        tch_xf_l2 = torch.norm(tch_xf, dim=(1,2)).reshape(-1,1,1)
        stu_xf_l2 = torch.norm(stu_xf, dim=(1,2)).reshape(-1,1,1)

        # tch_xf_l2 = (tch_xf*tch_xf).sum(dim=2).sum(dim=1).reshape(-1,1,1)
        # stu_xf_l2 = (stu_xf*stu_xf).sum(dim=2).sum(dim=1).reshape(-1,1,1)

        diff = tch_xf / tch_xf_l2 - stu_xf / stu_xf_l2
        diff_l2 = torch.norm(diff, dim=(1,2)).sum()
        l2_losses.append(diff_l2)

        # diff = tch_x - stu_x
        # diff_l2 = diff*diff
        # out = ME.MinkowskiGlobalSumPooling()(diff_l2)
        # l2_losses.append(out.F.sum(-1))

    l2_loss = torch.stack(l2_losses).sum() / len(l2_losses)
    # print(l2_loss)

    return l2_loss

def train_distill(model, data_loader, val_data_loader, config, transform_data_fn=None):
    '''
    the distillation training
    some cfgs here
    '''

    # distill_lambda = 1
    # distill_lambda = 0.33
    distill_lambda = 0.67

    # TWO_STAGE=True: Transformer is first trained with L2 loss to match ResNet's activation, and then it fintunes like normal training on the second stage. 
    # TWO_STAGE=False: Transformer trains with combined loss

    TWO_STAGE = False
    # STAGE_PERCENTAGE = 0.7

    device = get_torch_device(config.is_cuda)
    # Set up the train flag for batch normalization
    model.train()

    # Configuration
    data_timer, iter_timer = Timer(), Timer()
    data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
    losses, scores = AverageMeter(), AverageMeter()

    optimizer = initialize_optimizer(model.parameters(), config)
    scheduler = initialize_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)

    # Train the network
    logging.info('===> Start training')
    best_val_miou, best_val_iter, curr_iter, epoch, is_training = 0, 0, 1, 1, True

    # TODO: 
    # load the sub-model only
    # FIXME: some dirty hard-written stuff, only supporting current state

    tch_model_cls = load_model('Res16UNet18A')
    tch_model = tch_model_cls(3,20,config).to(device)

    # checkpoint_fn = "/home/zhaotianchen/project/point-transformer/SpatioTemporalSegmentation-ScanNet/outputs/ScannetSparseVoxelizationDataset/Res16UNet18A/resnet_base/weights.pth"
    checkpoint_fn = "/home/zhaotianchen/project/point-transformer/SpatioTemporalSegmentation-ScanNet/outputs/ScannetSparseVoxelizationDataset/Res16UNet18A/Res18A/weights.pth" # voxel-size: 0.05
    assert osp.isfile(checkpoint_fn)
    logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
    state = torch.load(checkpoint_fn)
    d = {k:v for k,v in state['state_dict'].items() if 'map' not in k }
    tch_model.load_state_dict(d)
    if 'best_val' in state:
        best_val_miou = state['best_val']
        best_val_iter = state['best_val_iter']
    logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))

    if config.resume:
        raise NotImplementedError
        # Test loaded ckpt first

        # checkpoint_fn = config.resume + '/weights.pth'
        # if osp.isfile(checkpoint_fn):
            # logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            # state = torch.load(checkpoint_fn)
            # curr_iter = state['iteration'] + 1
            # epoch = state['epoch']
            # d = {k:v for k,v in state['state_dict'].items() if 'map' not in k }
            # model.load_state_dict(d)
            # if config.resume_optimizer:
                # scheduler = initialize_scheduler(optimizer, config, last_step=curr_iter)
                # optimizer.load_state_dict(state['optimizer'])
            # if 'best_val' in state:
                # best_val_miou = state['best_val']
                # best_val_iter = state['best_val_iter']
            # logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
        # else:
            # raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

    # test after loading the ckpt
    v_loss, v_score, v_mAP, v_mIoU = test(tch_model, val_data_loader, config)
    logging.info('Tch model tested, bes_miou: {}'.format(v_mIoU))

    data_iter = data_loader.__iter__()
    while is_training:

        num_class = 20
        total_correct_class = torch.zeros(num_class, device=device)
        total_iou_deno_class = torch.zeros(num_class, device=device)

        total_iteration = len(data_loader) // config.iter_size
        for iteration in range(total_iteration):

            # NOTE: for single stage distillation, L2 loss might be too large at first 
            # so we added a warmup training that don't use L2 loss
            if iteration < 0:
                use_distill = False
            else:
                use_distill = True

            # Stage 1 / Stage 2 boundary
            if TWO_STAGE:
                stage_boundary = int(total_iteration * STAGE_PERCENTAGE)

            optimizer.zero_grad()
            data_time, batch_loss = 0, 0
            iter_timer.tic()

            for sub_iter in range(config.iter_size):
                # Get training data
                data_timer.tic()
                if config.return_transformation:
                    coords, input, target, _, _, pointcloud, transformation = data_iter.next()
                else:
                    coords, input, target, _, _ = data_iter.next()  # ignore unique_map and inverse_map

                if config.use_aux:
                    assert target.shape[1] == 2
                    aux = target[:,1]
                    target = target[:,0]
                else:
                    aux = None

                # For some networks, making the network invariant to even, odd coords is important
                coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

                # Preprocess input
                if config.normalize_color:
                    input[:, :3] = input[:, :3] / 255. - 0.5
                    coords_norm = coords[:,1:] / coords[:,1:].max() - 0.5

                # cat xyz into the rgb feature
                if config.xyz_input:
                    input = torch.cat([coords_norm, input], dim=1)

                sinput = SparseTensor(input, coords, device=device)

                # TODO: return both-models
                # in order to not breaking the valid interface, use a get_loss to get the regsitered loss

                data_time += data_timer.toc(False)
                # model.initialize_coords(*init_args)
                if aux is not None:
                    raise NotImplementedError
                
                # flatten ground truth tensor
                target = target.view(-1).long().to(device)
                
                if TWO_STAGE:
                    if iteration < stage_boundary:
                        # Stage 1: train transformer on L2 loss
                        soutput, anchor = model(sinput, save_anchor=True)
                        # Make sure gradient don't flow to teacher model
                        with torch.no_grad():
                            _, tch_anchor = tch_model(sinput, save_anchor=True)
                        loss = DistillLoss(tch_anchor, anchor)
                    else:
                        # Stage 2: finetune transformer on Cross-Entropy
                        soutput = model(sinput)
                        loss = criterion(soutput.F, target.long())
                else:
                    if use_distill: # after warm up 
                        soutput, anchor = model(sinput, save_anchor=True)
                        # if pretrained teacher, do not let the grad flow to teacher to update its params
                        with torch.no_grad():
                            tch_soutput, tch_anchor = tch_model(sinput, save_anchor=True)

                    else: # warming up
                        soutput = model(sinput)
                    # The output of the network is not sorted
                    loss = criterion(soutput.F, target.long())
                    #  Add L2 loss if use distillation
                    if use_distill:
                        distill_loss = DistillLoss(tch_anchor, anchor)*distill_lambda
                        loss += distill_loss

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
            score = precision_at_one(pred, target, ignore_label=-1)
            losses.update(batch_loss, target.size(0))
            scores.update(score, target.size(0))

            # calc the train-iou
            for l in range(num_class):
                total_correct_class[l] += ((pred == l) & (target == l)).sum()
                total_iou_deno_class[l] += (((pred == l) & (target!=-1)) | (target == l) ).sum()

            if curr_iter >= config.max_iter:
                is_training = False
                break

            if curr_iter % config.stat_freq == 0 or curr_iter == 1:
                lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
                debug_str = "[{}] ===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(config.log_dir,
                        epoch, curr_iter,
                        len(data_loader) // config.iter_size, losses.avg, lrs)
                debug_str += "Score {:.3f}\tData time: {:.4f}, Iter time: {:.4f}".format(
                        scores.avg, data_time_avg.avg, iter_time_avg.avg)
                logging.info(debug_str)
                if use_distill and not TWO_STAGE:
                    logging.info('Loss {} Distill Loss:{}'.format(loss, distill_loss))
                # Reset timers
                data_time_avg.reset()
                iter_time_avg.reset()
                losses.reset()
                scores.reset()

            # Save current status, save before val to prevent occational mem overflow
            if curr_iter % config.save_freq == 0:
                checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, save_inter=True)

            # Validation
            if curr_iter % config.val_freq == 0:
                val_miou = validate(model, val_data_loader, None, curr_iter, config, transform_data_fn)
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

        IoU = (total_correct_class) / (total_iou_deno_class+1e-6)
        logging.info('train point avg class IoU: %f' % ((IoU).mean()*100.))

        epoch += 1

    # Explicit memory cleanup
    if hasattr(data_iter, 'cleanup'):
        data_iter.cleanup()

    # Save the final model
    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)
    v_loss, v_score, v_mAP, val_miou = test(model, val_data_loader, config)
    if val_miou > best_val_miou:
        best_val_miou = val_miou
        best_val_iter = curr_iter
        checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, "best_val")
    logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))


