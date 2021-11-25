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
        # return average_precision_score(label, prob_np)
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
        save_dict['coord'] = []

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

        # Calc of the iou
        total_correct = np.zeros(num_labels)
        total_seen = np.zeros(num_labels)
        total_positive = np.zeros(num_labels)

        for iteration in range(max_iter):
            data_timer.tic()
            if config.return_transformation:
                coords, input, target, unique_map_list, inverse_map_list, pointcloud, transformation = data_iter.next()
            else:
                coords, input, target, unique_map_list, inverse_map_list = data_iter.next()
            data_time = data_timer.toc(False)

            if config.use_aux:
                assert target.shape[1] == 2
                aux = target[:,1]
                target = target[:,0]
            else:
                aux = None

            # Preprocess input
            iter_timer.tic()

            if config.normalize_color:
                input[:, :3] = input[:, :3] / input[:,:3].max() - 0.5
                coords_norm = coords[:,1:] / coords[:,1:].max() - 0.5

            XYZ_INPUT = config.xyz_input
            # cat xyz into the rgb feature
            if XYZ_INPUT:
                input = torch.cat([coords_norm, input], dim=1)

            sinput = ME.SparseTensor(input, coords, device=device)

            # Feed forward
            if aux is not None:
                soutput = model(sinput)
            else:
                soutput = model(sinput, iter_ = iteration / max_iter, enable_point_branch=config.enable_point_branch)
            output = soutput.F
            if torch.isnan(output).sum() > 0:
                import ipdb; ipdb.set_trace()

            pred = get_prediction(dataset, output, target).int()
            assert sum([int(t.shape[0]) for t in unique_map_list]) == len(pred), "number of points in unique_map doesn't match predition, do not enable preprocessing"
            iter_time = iter_timer.toc(False)

            if config.save_pred:
                # troublesome processing for splitting each batch's data, and export
                batch_ids = sinput.C[:,0]
                splits_at = torch.stack([torch.where(batch_ids == i)[0][-1] for i in torch.unique(batch_ids)]).int()
                splits_at = splits_at + 1
                splits_at_leftshift_one = splits_at.roll(shifts=1)
                splits_at_leftshift_one[0] = 0
                # len_per_batch = splits_at - splits_at_leftshift_one

                len_sum = 0
                batch_id = 0
                for start, end in zip(splits_at_leftshift_one, splits_at):
                    len_sum += len(pred[int(start):int(end)])
                    pred_this_batch = pred[int(start):int(end)]
                    coord_this_batch = pred[int(start):int(end)]
                    save_dict['pred'].append(pred_this_batch[inverse_map_list[batch_id]])
                    # save_dict['coord'].append(coord_this_batch[inverse_map_list[batch_id]])
                    batch_id += 1
                assert len_sum == len(pred)

            # Unpack it to original length
            REVERT_WHOLE_POINTCLOUD = True
            print('{}/{}'.format(iteration, max_iter))
            if REVERT_WHOLE_POINTCLOUD:
                whole_pred = []
                whole_target = []
                for batch_ in range(config.batch_size):
                    batch_mask_ = (soutput.C[:,0] == batch_).cpu().numpy()
                    if batch_mask_.sum() == 0: # for empty batch, skip em 
                        continue
                    try:
                        whole_pred_ = soutput.F[batch_mask_][inverse_map_list[batch_]]
                    except:
                        import ipdb; ipdb.set_trace()
                    whole_target_ = target[batch_mask_][inverse_map_list[batch_]]
                    whole_pred.append(whole_pred_)
                    whole_target.append(whole_target_)
                whole_pred = torch.cat(whole_pred, dim=0)
                whole_target = torch.cat(whole_target, dim=0)

                pred = get_prediction(dataset, whole_pred, whole_target).int()
                output = whole_pred
                target = whole_target

            if has_gt:
                target_np = target.numpy()
                num_sample = target_np.shape[0]
                target = target.to(device)
                output = output.to(device)

                cross_ent = criterion(output, target.long())
                losses.update(float(cross_ent), num_sample)
                scores.update(precision_at_one(pred, target), num_sample)
                hist += fast_hist(pred.cpu().numpy().flatten(), target_np.flatten(), num_labels) # within fast hist, mark label should >=0 & < num_label to filter out 255 / -1
                ious = per_class_iu(hist) * 100
                prob = torch.nn.functional.softmax(output, dim=-1)

                pred = pred[target != -1]
                target = target[target != -1]

                # for _ in range(num_labels): # debug for SemKITTI: spvnas way of calc miou
                    # total_seen[_] += torch.sum(target == _)
                    # total_correct[_] += torch.sum((pred == target) & (target == _))
                    # total_positive[_] += torch.sum(pred == _)

                # ious_ = []
                # for _ in range(num_labels):
                    # if total_seen[_] == 0:
                        # ious_.append(1)
                    # else:
                        # ious_.append(total_correct[_]/(total_seen[_] + total_positive[_] - total_correct[_]))
                # ious_ = torch.stack(ious_, dim=-1).cpu().numpy()*100
                # print(np.nanmean(per_class_iu(hist)), np.nanmean(ious_))
                # ious = np.array(ious_)*100

                # skip calculating aps
                ap = average_precision(prob.cpu().detach().numpy(), target_np)
                aps = np.vstack((aps, ap))
                # Due to heavy bias in class, there exists class with no test label at all
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ap_class = np.nanmean(aps, 0) * 100.

            if iteration % config.test_stat_freq == 0 and iteration > 0:
                reordered_ious = dataset.reorder_result(ious)
                reordered_ap_class = dataset.reorder_result(ap_class)
                # dirty fix for semnaticcKITTI has no getclassnames
                if hasattr(dataset, "class_names"):
                    class_names = dataset.get_classnames()
                else: # semnantic KITTI
                    class_names = None
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
        # torch.save(save_dict, os.path.join(config.log_dir, 'preds_{}_with_coord.pth'.format(split)))
        torch.save(save_dict, os.path.join(config.log_dir, 'preds_{}.pth'.format(split)))
        print("===> saved prediction result")

    global_time = global_timer.toc(False)

    save_map(model, config)

    reordered_ious = dataset.reorder_result(ious)
    reordered_ap_class = dataset.reorder_result(ap_class)
    if hasattr(dataset, "class_names"):
        class_names = dataset.get_classnames()
    else:
        class_names = None
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

# def test_points(model,
                                 # data_loader,
                                 # config,
                                 # with_aux=False,
                                 # save_dir=None,
                                 # split='eval',
                                 # use_voxel=True):
        # '''
        # :param pn_list: sn (list => int), the number of points in a scene
        # :param scene_list: sn (list => str), scene id
        # '''

        # pn_list = data_loader.dataset.point_num
        # scene_list = data_loader.dataset.scene_list
        # SEM_LABELS = data_loader.dataset.semantic_labels_list

        # model.eval()
        # total_seen = 0
        # total_correct = 0
        # total_seen_class = [0] * NUM_CLASSES
        # total_correct_class = [0] * NUM_CLASSES
        # total_iou_deno_class = [0] * NUM_CLASSES

        # if save_dir is not None:
                # save_dict = {}
                # save_dict['pred'] = []

        # use_voxel = not config.pure_point

        # scene_num = len(scene_list)
        # for scene_index in range(scene_num):
                # logging.info(' ======= {}/{} ======= '.format(scene_index, scene_num))
                # # scene_index = 0
                # scene_id = scene_list[scene_index]
                # point_num = pn_list[scene_index]
                # predict = np.zeros((point_num, NUM_CLASSES), dtype=np.float32) # pn,21
                # vote_num = np.zeros((point_num, 1), dtype=np.int) # pn,1
                # for idx, batch_data in enumerate(data_loader):
                        # # logging.info('batch {}'.format(idx))
                        # if with_aux:
                                # pc, seg, aux, smpw, pidx= batch_data
                                # aux = aux.cuda()
                                # seg = seg.cuda()
                        # else:
                                # pc, seg, smpw, pidx= batch_data
                        # if pidx.max() > point_num:
                                # import ipdb; ipdb.set_trace()

                        # pc = pc.cuda().float()
                        # '''
                        # use voxel-forward for testing the scannet
                        # '''
                        # if use_voxel:
                                # coords = torch.unbind(pc[:,:,:3]/config.voxel_size, dim=0)
                                # # Normalize the xyz after the coord is set
                                # # pc[:,:,:3] = pc[:,:,:3] / pc[:,:,:3].mean()
                                # feats = torch.unbind(pc[:,:,:], dim=0) # use all 6 chs for eval
                                # coords, feats= ME.utils.sparse_collate(coords, feats) # the returned coords adds a batch-dimw
                                # pc = ME.TensorField(features=feats.float(),coordinates=coords.cuda()) # [xyz, norm_xyz, rgb]
                                # voxels = pc.sparse()
                                # seg = seg.view(-1)
                                # inputs = voxels

                        # else:
                                # # DEBUG: discrete input xyz for point-based method
                                # feats = torch.unbind(pc[:,:,:], dim=0)
                                # coords = torch.unbind(pc[:,:,:3]/config.voxel_size, dim=0)
                                # coords, feats= ME.utils.sparse_collate(coords, feats) # the returned coords adds a batch-dim

                                # pc = ME.TensorField(features=feats.float(),coordinates=coords.cuda()) # [xyz, norm_xyz, rgb]
                                # voxels = pc.sparse()
                                # pc_ = voxels.slice(pc)
                                # # pc = torch.cat([pc_.C[:,1:],pc_.F[:,:3:]],dim=1).reshape([-1, config.num_points, 6])
                                # pc = pc_.F.reshape([-1, config.num_points, 6])

                                # # discrete_coords = coords.reshape([-1, config.num_points, 4])[:,:,1:] # the batch does not have drop-last
                                # # pc[:,:,:3] = discrete_coords

                                # pc = pc.transpose(1,2)
                                # inputs = pc

                        # if with_aux:
                                # # DEBUG: Use target as instance for now
                                # pred = model(inputs, instance=aux) # B,N,C
                        # else:
                                # pred = model(inputs) # B,N,C

                        # if use_voxel:
                                # assert isinstance(pred, ME.SparseTensor)
                                # pred = pred.slice(pc).F
                                # try:
                                        # pred = pred.reshape([-1, config.num_points, NUM_CLASSES])       # leave the 1st dim, since no droplast
                                # except RuntimeError:
                                        # import ipdb; ipdb.set_trace()

                        # pred = torch.nn.functional.softmax(pred, dim=2)
                        # pred = pred.cpu().detach().numpy()

                        # pidx = pidx.numpy() # B,N
                        # predict, vote_num = vote(predict, vote_num, pred, pidx)

                # predict = predict / vote_num

                # if save_dir is not None:
                        # if np.isnan(predict).any():
                                # print("found nan in scene{}".format(scene_id))
                                # import ipdb; ipdb.set_trace()
                        # save_dict['pred'].append(np.argmax(predict, axis=-1))

                # # predict = np.argmax(predict[:, 1:], axis=-1) # pn  # debug WHY?
                # predict = np.argmax(predict, axis=-1) # pn
                # labels = SEM_LABELS[scene_index]

                # '''
                # additional logic for handling 20 class output
                # '''
                # labels = labels - 1
                # correct = predict == labels
                # correct = correct[labels != -1]

                # total_seen += np.sum(labels.size) # point_num
                # # total_correct += np.sum((predict == labels) & (labels > 0))
                # total_correct += np.sum(correct)
                # logging.info('accuracy:{} '.format(total_correct / total_seen))
                # for l in range(NUM_CLASSES):
                        # total_seen_class[l] += np.sum((labels == l) & (labels >= 0))
                        # total_correct_class[l] += np.sum((predict == l) & (labels == l))
                        # total_iou_deno_class[l] += np.sum(((predict == l) & (labels >= 0)) | (labels == l))

                # '''Uncomment this to save the map, this could take about 500M sapce'''
                # # save_map(model, config)
                # # import ipdb; ipdb.set_trace()

        # # final save
        # if save_dir is not None:
                # torch.save(save_dict, os.path.join(save_dir,'{}_pred.pth'.format(split)))

        # IoU = np.array(total_correct_class)/(np.array(total_iou_deno_class,dtype=np.float)+1e-6)
        # logging.info('eval point avg class IoU: %f' % (np.mean(IoU)))
        # IoU_Class = 'Each Class IoU:::\n'
        # for i in range(IoU.shape[0]):
                # logging.info('Class %d : %.4f'%(i+1, IoU[i]))
        # logging.info('eval accuracy: %f'% (total_correct / float(total_seen)))
        # logging.info('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/(np.array(total_seen_class,dtype=np.float)+1e-6))))





