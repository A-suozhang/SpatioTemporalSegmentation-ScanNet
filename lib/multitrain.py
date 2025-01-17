# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import logging
import os.path as osp
import os
import sys

import torch
from torch import nn

from lib.test import test
from lib.utils import checkpoint, precision_at_one, \
        Timer, AverageMeter, get_prediction, get_torch_device
from lib.solvers import initialize_optimizer, initialize_scheduler

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

import numpy as np

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader

def validate(model, val_data_loader, writer, curr_iter, config, transform_data_fn=None):
    v_loss, v_score, v_mAP, v_mIoU = test(model, val_data_loader, config)
    return v_mIoU

def setup_logger(config):
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


def train(NetClass, data_loader, val_data_loader, config, transform_data_fn=None):
    num_devices = torch.cuda.device_count()
    print("Running on " + str(num_devices) + " GPUs. Total batch size: " + str(num_devices * config.batch_size))
    config.world_size = num_devices
    mp.spawn(train_worker, nprocs=num_devices, args=(num_devices, NetClass, data_loader, val_data_loader, config, transform_data_fn))

def train_worker(gpu, num_devices, NetClass, data_loader, val_data_loader, config, transform_data_fn=None):
    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))
        rank = gpu
    addr = 23491
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:{}".format(addr),
        world_size=num_devices,
        rank=rank
    )

    # replace with DistributedSampler
    if config.multiprocess:
        from lib.dataloader_dist import InfSampler
        sampler = InfSampler(data_loader.dataset)
        data_loader = DataLoader(
            dataset=data_loader.dataset,
            num_workers=data_loader.num_workers,
            batch_size=data_loader.batch_size,
            collate_fn=data_loader.collate_fn,
            worker_init_fn=data_loader.worker_init_fn,
            sampler=sampler
        )

    if data_loader.dataset.NUM_IN_CHANNEL is not None:
        num_in_channel = data_loader.dataset.NUM_IN_CHANNEL
    else:
        num_in_channel = 3
    num_labels = data_loader.dataset.NUM_LABELS

    # load model
    if config.pure_point:
        model = NetClass(num_class=config.num_labels, N=config.num_points, normal_channel=config.num_in_channel)
    else:
        if config.model == 'MixedTransformer':
            model = NetClass(config, num_class=num_labels,N=config.num_points,normal_channel=num_in_channel)
        elif config.model == 'MinkowskiVoxelTransformer':
            model = NetClass(config, num_in_channel, num_labels)
        elif config.model == 'MinkowskiTransformerNet':
            model = NetClass(config, num_in_channel, num_labels)
        elif "Res" in config.model:
            model = NetClass(num_in_channel, num_labels, config)
        else:
            model = NetClass(num_in_channel, num_labels, config)

    if config.weights == 'modelzoo':
        model.preload_modelzoo()
    elif config.weights.lower() != 'none':
        state = torch.load(config.weights)
        # delete the keys containing the attn since it raises size mismatch
        d = { k:v for k, v in state['state''_dict'].items() if 'map' not in k }
        if config.weights_for_inner_model:
            model.model.load_state_dict(d)
        else:
            if config.lenient_weight_loading:
                matched_weights = load_state_with_same_shape(model, state['state_dict'])
                model_dict = model.state_dict()
                model_dict.update(matched_weights)
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(d, strict=False)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    # use model with DDP
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)
    # Synchronized batch norm
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

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
    if rank == 0:
        setup_logger(config)
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
    device = gpu # multitrain fed in the device
    if config.dataset == "SemanticKITTI":
        num_class = 19
        config.normalize_color = False
        config.xyz_input = False
        val_freq_ = config.val_freq
        config.val_freq = config.val_freq*10 # origianl val_freq_
    elif config.dataset == 'S3DIS':
        num_class = 13
        config.normalize_color = False
        config.xyz_input = False
        val_freq_ = config.val_freq
    elif config.dataset == "Nuscenes":
        num_class = 16
        config.normalize_color = False
        config.xyz_input = False
        val_freq_ = config.val_freq
        config.val_freq = config.val_freq*50
    else:
        val_freq_ = config.val_freq
        num_class = 20

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
                    input[:, :3] = input[:, :3] / input[:,:3].max() - 0.5
                    coords_norm = coords[:,1:] / coords[:,1:].max() - 0.5

                # cat xyz into the rgb feature
                if config.xyz_input:
                    input = torch.cat([coords_norm, input], dim=1)
                # print(device)

                sinput = SparseTensor(input, coords, device=device)

                # d = {}
                # d['coord'] = sinput.C
                # d['feat'] = sinput.F
                # torch.save(d, 'voxel.pth')
                # import ipdb; ipdb.set_trace()

                data_time += data_timer.toc(False)
                # model.initialize_coords(*init_args)
                if aux is not None:
                    soutput = model(sinput, aux)
                elif config.enable_point_branch:
                    soutput = model(sinput, iter_= curr_iter / config.max_iter, enable_point_branch=True)
                else:
                    soutput = model(sinput, iter_= curr_iter / config.max_iter)  # feed in the progress of training for annealing inside the model
                    # soutput = model(sinput)
                # The output of the network is not sorted
                target = target.view(-1).long().to(device)

                loss = criterion(soutput.F, target.long())

                # ====== other loss regs =====
                cur_loss = torch.tensor([0.], device=device)
                if hasattr(model, 'module.block1'):
                    cur_loss = torch.tensor([0.], device=device)

                    if hasattr(model.module.block1[0],'vq_loss'):
                        if model.block1[0].vq_loss is not None:
                            cur_loss = torch.tensor([0.], device=device)
                            for n, m in model.named_children():
                                if 'block' in n:
                                    cur_loss += m[0].vq_loss # m is the nn.Sequential obj, m[0] is the TRBlock
                            logging.info('Cur Loss: {}, Cur vq_loss: {}'.format(loss, cur_loss))
                            loss += cur_loss

                    if hasattr(model.module.block1[0],'diverse_loss'):
                        if model.block1[0].diverse_loss is not None:
                            cur_loss = torch.tensor([0.], device=device)
                            for n, m in model.named_children():
                                if 'block' in n:
                                    cur_loss += m[0].diverse_loss # m is the nn.Sequential obj, m[0] is the TRBlock
                            logging.info('Cur Loss: {}, Cur diverse _loss: {}'.format(loss, cur_loss))
                            loss += cur_loss

                    if hasattr(model.module.block1[0],'label_reg'):
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
                if not config.use_sam:
                    loss.backward()
                else:
                    with model.no_sync():
                        loss.backward()

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
                else:
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
                lrs = ', '.join(['{:.3e}'.format(g['lr']) for g in optimizer.param_groups])
                IoU = ((total_correct_class) / (total_iou_deno_class+1e-6)).mean()*100.
                debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
                        epoch, curr_iter,
                        len(data_loader) // config.iter_size, losses.avg, lrs)
                debug_str += "Score {:.3f}\tIoU {:.3f}\tData time: {:.4f}, Iter time: {:.4f}".format(
                        scores.avg, IoU.item(), data_time_avg.avg, iter_time_avg.avg)
                if regs.avg > 0:
                    debug_str += "\n Additional Reg Loss {:.3f}".format(regs.avg)

                if rank == 0:
                    logging.info(debug_str)
                # Reset timers
                data_time_avg.reset()
                iter_time_avg.reset()
                # Write logs
                losses.reset()
                scores.reset()

            # only save status on the 1st gpu
            if rank == 0:

                # Save current status, save before val to prevent occational mem overflow
                if curr_iter % config.save_freq == 0:
                    checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, save_inter=True)

                # Validation
                if curr_iter % config.val_freq == 0:
                    val_miou = validate(model, val_data_loader, None, curr_iter, config, transform_data_fn) # feedin None for SummaryWriter args
                    if val_miou > best_val_miou:
                        best_val_miou = val_miou
                        best_val_iter = curr_iter
                        checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter,
                                             "best_val", save_inter=True)
                    if rank == 0:
                        logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

                    # Recover back
                    model.train()

            # End of iteration
            curr_iter += 1

        IoU = (total_correct_class) / (total_iou_deno_class+1e-6)
        if rank == 0:
            logging.info('train point avg class IoU: %f' % ((IoU).mean()*100.))

        epoch += 1

    # Explicit memory cleanup
    if hasattr(data_iter, 'cleanup'):
        data_iter.cleanup()

    # Save the final model
    if rank == 0:
        checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)
        v_loss, v_score, v_mAP, val_mIoU = test(model, val_data_loader, config)

        if val_miou > best_val_miou and rank == 0:
            best_val_miou = val_miou
            best_val_iter = curr_iter
            logging.info("Final best miou: {}  at iter {} ".format(val_miou, curr_iter))
            checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, "best_val")

            logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))
    # print("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))


# def train_point(model, data_loader, val_data_loader, config, transform_data_fn=None):

    # device = get_torch_device(config.is_cuda)
    # # Set up the train flag for batch normalization
    # model.train()

    # # Configuration
    # writer = SummaryWriter(log_dir=config.log_dir)
    # data_timer, iter_timer = Timer(), Timer()
    # data_time_avg, iter_time_avg = AverageMeter(), AverageMeter()
    # losses, scores = AverageMeter(), AverageMeter()

    # optimizer = initialize_optimizer(model.parameters(), config)
    # scheduler = initialize_scheduler(optimizer, config)
    # criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # writer = SummaryWriter(log_dir=config.log_dir)

    # # Train the network
    # logging.info('===> Start training')
    # best_val_miou, best_val_iter, curr_iter, epoch, is_training = 0, 0, 1, 1, True

    # if config.resume:
        # checkpoint_fn = config.resume + '/weights.pth'
        # if osp.isfile(checkpoint_fn):
            # logging.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            # state = torch.load(checkpoint_fn)
            # curr_iter = state['iteration'] + 1
            # epoch = state['epoch']
            # model.load_state_dict(state['state_dict'])
            # if config.resume_optimizer:
                # scheduler = initialize_scheduler(optimizer, config, last_step=curr_iter)
                # optimizer.load_state_dict(state['optimizer'])
            # if 'best_val' in state:
                # best_val_miou = state['best_val']
                # best_val_iter = state['best_val_iter']
            # logging.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_fn, state['epoch']))
        # else:
            # raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

    # data_iter = data_loader.__iter__()
    # while is_training:

        # num_class = 20
        # total_correct_class = torch.zeros(num_class, device=device)
        # total_iou_deno_class = torch.zeros(num_class, device=device)

        # for iteration in range(len(data_loader) // config.iter_size):
            # optimizer.zero_grad()
            # data_time, batch_loss = 0, 0
            # iter_timer.tic()
            # for sub_iter in range(config.iter_size):
                # # Get training data
                # data= data_iter.next()
                # points, target, sample_weight = data
                # if config.pure_point:

                        # sinput = points.transpose(1,2).cuda().float()

                        # # DEBUG: use the discrete coord for point-based
                        # feats = torch.unbind(points[:,:,3:], dim=0)
                        # voxel_size = config.voxel_size
                        # coords = torch.unbind(points[:,:,:3]/voxel_size, dim=0)  # 0.05 is the voxel-size
                        # coords, feats= ME.utils.sparse_collate(coords, feats)
                        # # assert feats.reshape([16, 4096, -1]) == points[:,:,3:]
                        # points_ = ME.TensorField(features=feats.float(), coordinates=coords, device=device)
                        # tmp_voxel = points_.sparse()
                        # sinput_ = tmp_voxel.slice(points_)

                        # sinput = torch.cat([sinput_.C[:,1:], sinput_.F],dim=1).reshape([config.batch_size, config.num_points, 6])
                        # sinput = sinput.transpose(1,2).cuda().float()

                        # # sinput = torch.cat([coords[:,1:], feats],dim=1).reshape([config.batch_size, config.num_points, 6])
                        # # sinput = sinput.transpose(1,2).cuda().float()


                        # # For some networks, making the network invariant to even, odd coords is important
                        # # coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

                        # # Preprocess input
                        # # if config.normalize_color:
                            # # feats = feats / 255. - 0.5

                        # # torch.save(points[:,:,:3], './sandbox/tensorfield-c.pth')
                        # # torch.save(points_.C, './sandbox/points-c.pth')


                # else:
                        # # feats = torch.unbind(points[:,:,3:], dim=0) # WRONG: should also feed in xyz as inupt feature
                        # feats = torch.unbind(points[:,:,:], dim=0)
                        # voxel_size = config.voxel_size
                        # coords = torch.unbind(points[:,:,:3]/voxel_size, dim=0)  # 0.05 is the voxel-size
                        # coords, feats= ME.utils.sparse_collate(coords, feats)

                        # # For some networks, making the network invariant to even, odd coords is important
                        # # coords[:, 1:] += (torch.rand(3) * 100).type_as(coords)

                        # # Preprocess input
                        # # if config.normalize_color:
                            # # feats = feats / 255. - 0.5

                        # # they are the same
                        # points_ = ME.TensorField(features=feats.float(), coordinates=coords, device=device)
                        # # points_1 = ME.TensorField(features=feats.float(), coordinates=coords, device=device, quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
                        # # points_2 = ME.TensorField(features=feats.float(), coordinates=coords, device=device, quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE)
                        # sinput = points_.sparse()

                # data_time += data_timer.toc(False)
                # B, npoint = target.shape

                # # model.initialize_coords(*init_args)
                # soutput = model(sinput)
                # if config.pure_point:
                        # soutput = soutput.reshape([B*npoint, -1])
                # else:
                        # soutput = soutput.slice(points_).F
                        # # s1 = soutput.slice(points_)
                        # # print(soutput.quantization_mode)
                        # # soutput.quantization_mode = ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE
                        # # s2 = soutput.slice(points_)

                # # The output of the network is not sorted
                # target = (target-1).view(-1).long().to(device)

                # # catch NAN
                # if torch.isnan(target).sum() > 0:
                        # import ipdb; ipdb.set_trace()

                # loss = criterion(soutput, target)

                # if torch.isnan(loss).sum()>0:
                        # import ipdb; ipdb.set_trace()

                # loss = (loss*sample_weight.to(device)).mean()

                # # Compute and accumulate gradient
                # loss /= config.iter_size
                # batch_loss += loss.item()
                # loss.backward()

            # # Update number of steps
            # optimizer.step()
            # scheduler.step()

            # # CLEAR CACHE!
            # torch.cuda.empty_cache()


            # data_time_avg.update(data_time)
            # iter_time_avg.update(iter_timer.toc(False))

            # pred = get_prediction(data_loader.dataset, soutput, target)
            # score = precision_at_one(pred, target, ignore_label=-1)
            # losses.update(batch_loss, target.size(0))
            # scores.update(score, target.size(0))

            # # Calc the iou
            # for l in range(num_class):
                # total_correct_class[l] += ((pred == l) & (target == l)).sum()
                # total_iou_deno_class[l] += (((pred == l) & (target >= 0)) | (target == l)).sum()

            # if curr_iter >= config.max_iter:
                # is_training = False
                # break

            # if curr_iter % config.stat_freq == 0 or curr_iter == 1:
                # lrs = ', '.join(['{:.3e}'.format(x) for x in scheduler.get_lr()])
                # debug_str = "===> Epoch[{}]({}/{}): Loss {:.4f}\tLR: {}\t".format(
                        # epoch, curr_iter,
                        # len(data_loader) // config.iter_size, losses.avg, lrs)
                # debug_str += "Score {:.3f}\tData time: {:.4f}, Iter time: {:.4f}".format(
                        # scores.avg, data_time_avg.avg, iter_time_avg.avg)
                # # logging.info(debug_str)
                # # Reset timers
                # data_time_avg.reset()
                # iter_time_avg.reset()
                # # Write logs
                # writer.add_scalar('training/loss', losses.avg, curr_iter)
                # writer.add_scalar('training/precision_at_1', scores.avg, curr_iter)
                # writer.add_scalar('training/learning_rate', scheduler.get_lr()[0], curr_iter)
                # losses.reset()
                # scores.reset()

            # # Save current status, save before val to prevent occational mem overflow
            # if curr_iter % config.save_freq == 0:
                # checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, save_inter=True)

            # # Validation
            # # if curr_iter % config.val_freq == 0:
                # # val_miou = test_points(model, val_data_loader, writer, curr_iter, config, transform_data_fn)
                # # if val_miou > best_val_miou:
                    # # best_val_miou = val_miou
                    # # best_val_iter = curr_iter
                    # # checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter,
                                         # # "best_val")
                # # logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

                # # # Recover back
                # # model.train()

            # # End of iteration
            # curr_iter += 1

        # IoU = (total_correct_class) / (total_iou_deno_class+1e-6)
        # # logging.info('eval point avg class IoU: %f' % ((IoU).mean()*100.))

        # epoch += 1


    # # Explicit memory cleanup
    # if hasattr(data_iter, 'cleanup'):
        # data_iter.cleanup()

    # # Save the final model
    # checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter)

    # test_points(model, val_data_loader, config)
    # # if val_miou > best_val_miou:
        # # best_val_miou = val_miou
        # # best_val_iter = curr_iter
        # # checkpoint(model, optimizer, epoch, curr_iter, config, best_val_miou, best_val_iter, "best_val")
    # logging.info("Current best mIoU: {:.3f} at iter {}".format(best_val_miou, best_val_iter))

