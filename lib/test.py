# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import logging
import warnings
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from lib.utils import Timer, AverageMeter, precision_at_one, fast_hist, per_class_iu, \
        get_prediction, get_torch_device, save_map

import MinkowskiEngine as ME


def print_info(iteration,
                             max_iteration,
                             data_time,
                             iter_time,
                             has_gt=False,
                             losses=None,
                             scores=None,
                             ious=None,
                             hist=None,
                             ap_class=None,
                             class_names=None):
    debug_str = "{}/{}: ".format(iteration + 1, max_iteration)
    debug_str += "Data time: {:.4f}, Iter time: {:.4f}".format(data_time, iter_time)

    if has_gt:
        acc = hist.diagonal() / hist.sum(1) * 100
        debug_str += "\tLoss {loss.val:.3f} (AVG: {loss.avg:.3f})\t" \
                "Score {top1.val:.3f} (AVG: {top1.avg:.3f})\t" \
                "mIOU {mIOU:.3f} mAP {mAP:.3f} mAcc {mAcc:.3f}\n".format(
                        loss=losses, top1=scores, mIOU=np.nanmean(ious),
                        mAP=np.nanmean(ap_class), mAcc=np.nanmean(acc))
        if class_names is not None:
            debug_str += "\nClasses: " + " ".join(class_names) + '\n'
        debug_str += 'IOU: ' + ' '.join('{:.03f}'.format(i) for i in ious) + '\n'
        debug_str += 'mAP: ' + ' '.join('{:.03f}'.format(i) for i in ap_class) + '\n'
        debug_str += 'mAcc: ' + ' '.join('{:.03f}'.format(i) for i in acc) + '\n'

    logging.info(debug_str)


def average_precision(prob_np, target_np):
    num_class = prob_np.shape[1]
    label = label_binarize(target_np, classes=list(range(num_class)))
    with np.errstate(divide='ignore', invalid='ignore'):
        return average_precision_score(label, prob_np, None)


def test(model, data_loader, config, transform_data_fn=None, has_gt=True, save_pred=False, split=None):
    device = get_torch_device(config.is_cuda)
    dataset = data_loader.dataset
    num_labels = dataset.NUM_LABELS
    global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_label)
    losses, scores, ious = AverageMeter(), AverageMeter(), 0
    aps = np.zeros((0, num_labels))
    hist = np.zeros((num_labels, num_labels))

    # some cfgs concerning the usage of instance-level information
    config.save_pred = save_pred
    if split is not None:
        assert save_pred
    if config.save_pred:
        save_dict = {}
        save_dict['pred'] = []

    logging.info('===> Start testing')

    global_timer.tic()
    data_iter = data_loader.__iter__()
    max_iter = len(data_loader)
    max_iter_unique = max_iter

    # Fix batch normalization running mean and std
    model.eval()

    # Clear cache (when run in val mode, cleanup training cache)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for iteration in range(max_iter):
            data_timer.tic()
            if config.return_transformation:
                coords, input, target, pointcloud, transformation = data_iter.next()
            else:
                coords, input, target = data_iter.next()
            data_time = data_timer.toc(False)

            # Preprocess input
            iter_timer.tic()

            if config.normalize_color:
                    input[:, :3] = input[:, :3] / 255. - 0.5
                    coords_norm = coords[:,1:] / coords[:,1:].max() - 0.5

            XYZ_INPUT = config.xyz_input
            # cat xyz into the rgb feature
            if XYZ_INPUT:
                    input = torch.cat([coords_norm, input], dim=1)

            sinput = ME.SparseTensor(input, coords, device=device)

            # Feed forward
            inputs = (sinput,)
            soutput = model(*inputs)
            output = soutput.F

            pred = get_prediction(dataset, output, target).int()
            iter_time = iter_timer.toc(False)

            if config.save_pred:

                batch_ids = sinput.C[:,0]
                splits_at = torch.stack([torch.where(batch_ids == i)[0][-1] for i in torch.unique(batch_ids)]).int()
                splits_at = splits_at + 1
                splits_at_leftshift_one = splits_at.roll(shifts=1)
                splits_at_leftshift_one[0] = 0
                # len_per_batch = splits_at - splits_at_leftshift_one

                len_sum = 0
                for start, end in zip(splits_at_leftshift_one, splits_at):
                    len_sum += len(pred[int(start):int(end)])
                    save_dict['pred'].append(pred[int(start):int(end)])
                assert len_sum == len(pred)

            if has_gt:
                target_np = target.numpy()
                num_sample = target_np.shape[0]
                target = target.to(device)

                cross_ent = criterion(output, target.long())
                losses.update(float(cross_ent), num_sample)
                scores.update(precision_at_one(pred, target), num_sample)
                hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels) # within fast hist, mark label should >=0 & < num_label to filter out 255 / -1
                ious = per_class_iu(hist) * 100

                prob = torch.nn.functional.softmax(output, dim=1)
                ap = average_precision(prob.cpu().detach().numpy(), target_np)
                aps = np.vstack((aps, ap))
                # Due to heavy bias in class, there exists class with no test label at all
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ap_class = np.nanmean(aps, 0) * 100.

            if iteration % config.test_stat_freq == 0 and iteration > 0:
                reordered_ious = dataset.reorder_result(ious)
                reordered_ap_class = dataset.reorder_result(ap_class)
                class_names = dataset.get_classnames()
                print_info(
                        iteration,
                        max_iter_unique,
                        data_time,
                        iter_time,
                        has_gt,
                        losses,
                        scores,
                        reordered_ious,
                        hist,
                        reordered_ap_class,
                        class_names=class_names)

            if iteration % 5 == 0:
                # Clear cache
                torch.cuda.empty_cache()

    if config.save_pred:
        torch.save(save_dict, os.path.join(config.log_dir, 'preds_{}.pth'.format(split)))

    global_time = global_timer.toc(False)

    reordered_ious = dataset.reorder_result(ious)
    reordered_ap_class = dataset.reorder_result(ap_class)
    class_names = dataset.get_classnames()
    print_info(
            iteration,
            max_iter_unique,
            data_time,
            iter_time,
            has_gt,
            losses,
            scores,
            reordered_ious,
            hist,
            reordered_ap_class,
            class_names=class_names)

    logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

    # Explicit memory cleanup
    if hasattr(data_iter, 'cleanup'):
        data_iter.cleanup()

    return losses.avg, scores.avg, np.nanmean(ap_class), np.nanmean(per_class_iu(hist)) * 100

# ===============================================================================================

NUM_CLASSES = 20

def load_checkpoint(model, filename):
        if os.path.isfile(filename):
                logging.info("==> Loading from checkpoint %s" % filename)
                checkpoint = torch.load(filename)
                epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model_state'])
                logging.info("==> Done")
        else:
                logging.info(filename)
                raise FileNotFoundError
        return epoch


def vote(predict, vote_num, pred, points_idx):
        ''' numpy array
        :param predict: (pn,21) float
        :param vote_num: (pn,1) int
        :param pred: (bs,np,21) float
        :param points_idx: (bs,np) int
        '''
        bs, np = points_idx.shape
        for i in range(bs):
                for j in range(np):
                        pred_ = pred[i, j, :] # 21
                        pidx_ = points_idx[i, j] # int
                        predict[pidx_, :] += pred_
                        vote_num[pidx_, 0] += 1
        return predict, vote_num

def test_points(model,
                                 data_loader,
                                 config,
                                 with_aux=False,
                                 save_dir=None,
                                 split='eval',
                                 use_voxel=True):
        '''
        :param pn_list: sn (list => int), the number of points in a scene
        :param scene_list: sn (list => str), scene id
        '''

        pn_list = data_loader.dataset.point_num
        scene_list = data_loader.dataset.scene_list
        SEM_LABELS = data_loader.dataset.semantic_labels_list

        model.eval()
        total_seen = 0
        total_correct = 0
        total_seen_class = [0] * NUM_CLASSES
        total_correct_class = [0] * NUM_CLASSES
        total_iou_deno_class = [0] * NUM_CLASSES

        if save_dir is not None:
                save_dict = {}
                save_dict['pred'] = []

        use_voxel = not config.pure_point

        scene_num = len(scene_list)
        for scene_index in range(scene_num):
                logging.info(' ======= {}/{} ======= '.format(scene_index, scene_num))
                # scene_index = 0
                scene_id = scene_list[scene_index]
                point_num = pn_list[scene_index]
                predict = np.zeros((point_num, NUM_CLASSES), dtype=np.float32) # pn,21
                vote_num = np.zeros((point_num, 1), dtype=np.int) # pn,1
                for idx, batch_data in enumerate(data_loader):
                        # logging.info('batch {}'.format(idx))
                        if with_aux:
                                pc, seg, aux, smpw, pidx= batch_data
                                aux = aux.cuda()
                                seg = seg.cuda()
                        else:
                                pc, seg, smpw, pidx= batch_data
                        if pidx.max() > point_num:
                                import ipdb; ipdb.set_trace()

                        pc = pc.cuda().float()
                        '''
                        use voxel-forward for testing the scannet
                        '''
                        if use_voxel:
                                coords = torch.unbind(pc[:,:,:3]/config.voxel_size, dim=0)
                                # Normalize the xyz after the coord is set
                                # pc[:,:,:3] = pc[:,:,:3] / pc[:,:,:3].mean()
                                feats = torch.unbind(pc[:,:,:], dim=0) # use all 6 chs for eval
                                coords, feats= ME.utils.sparse_collate(coords, feats) # the returned coords adds a batch-dimw
                                pc = ME.TensorField(features=feats.float(),coordinates=coords.cuda()) # [xyz, norm_xyz, rgb]
                                voxels = pc.sparse()
                                seg = seg.view(-1)
                                inputs = voxels

                        else:
                                # DEBUG: discrete input xyz for point-based method
                                feats = torch.unbind(pc[:,:,:], dim=0)
                                coords = torch.unbind(pc[:,:,:3]/config.voxel_size, dim=0)
                                coords, feats= ME.utils.sparse_collate(coords, feats) # the returned coords adds a batch-dim

                                pc = ME.TensorField(features=feats.float(),coordinates=coords.cuda()) # [xyz, norm_xyz, rgb]
                                voxels = pc.sparse()
                                pc_ = voxels.slice(pc)
                                # pc = torch.cat([pc_.C[:,1:],pc_.F[:,:3:]],dim=1).reshape([-1, config.num_points, 6])
                                pc = pc_.F.reshape([-1, config.num_points, 6])

                                # discrete_coords = coords.reshape([-1, config.num_points, 4])[:,:,1:] # the batch does not have drop-last
                                # pc[:,:,:3] = discrete_coords

                                pc = pc.transpose(1,2)
                                inputs = pc

                        if with_aux:
                                # DEBUG: Use target as instance for now
                                pred = model(inputs, instance=aux) # B,N,C
                        else:
                                pred = model(inputs) # B,N,C

                        if use_voxel:
                                assert isinstance(pred, ME.SparseTensor)
                                pred = pred.slice(pc).F
                                try:
                                        pred = pred.reshape([-1, config.num_points, NUM_CLASSES])       # leave the 1st dim, since no droplast
                                except RuntimeError:
                                        import ipdb; ipdb.set_trace()

                        pred = torch.nn.functional.softmax(pred, dim=2)
                        pred = pred.cpu().detach().numpy()

                        pidx = pidx.numpy() # B,N
                        predict, vote_num = vote(predict, vote_num, pred, pidx)

                predict = predict / vote_num

                if save_dir is not None:
                        if np.isnan(predict).any():
                                print("found nan in scene{}".format(scene_id))
                                import ipdb; ipdb.set_trace()
                        save_dict['pred'].append(np.argmax(predict, axis=-1))

                # predict = np.argmax(predict[:, 1:], axis=-1) # pn  # debug WHY?
                predict = np.argmax(predict, axis=-1) # pn
                labels = SEM_LABELS[scene_index]

                '''
                additional logic for handling 20 class output
                '''
                labels = labels - 1
                correct = predict == labels
                correct = correct[labels != -1]

                total_seen += np.sum(labels.size) # point_num
                # total_correct += np.sum((predict == labels) & (labels > 0))
                total_correct += np.sum(correct)
                logging.info('accuracy:{} '.format(total_correct / total_seen))
                for l in range(NUM_CLASSES):
                        total_seen_class[l] += np.sum((labels == l) & (labels >= 0))
                        total_correct_class[l] += np.sum((predict == l) & (labels == l))
                        total_iou_deno_class[l] += np.sum(((predict == l) & (labels >= 0)) | (labels == l))

                '''Uncomment this to save the map, this could take about 500M sapce'''
                # save_map(model, config)
                # import ipdb; ipdb.set_trace()

        # final save
        if save_dir is not None:
                torch.save(save_dict, os.path.join(save_dir,'{}_pred.pth'.format(split)))

        IoU = np.array(total_correct_class)/(np.array(total_iou_deno_class,dtype=np.float)+1e-6)
        logging.info('eval point avg class IoU: %f' % (np.mean(IoU)))
        IoU_Class = 'Each Class IoU:::\n'
        for i in range(IoU.shape[0]):
                logging.info('Class %d : %.4f'%(i+1, IoU[i]))
        logging.info('eval accuracy: %f'% (total_correct / float(total_seen)))
        logging.info('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/(np.array(total_seen_class,dtype=np.float)+1e-6))))




