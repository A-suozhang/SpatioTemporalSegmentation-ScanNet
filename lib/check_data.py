import torch
import MinkowskiEngine as ME

def check_data(model, train_data_loader, val_data_loader, config):
    data_iter = train_data_loader.__iter__()

    sample_size = 1
    # strides = [2,4,8,16]
    strides = [2]

    for stride in strides:

        coordinate_map_key = [stride]*3
        pool0 = ME.MinkowskiSumPooling(kernel_size=stride, stride=stride, dilation=1, dimension=3)
        all_neis_scenes = []
        for _ in range(sample_size):
            coords, input, target, _, _ = data_iter.next()

            assert coords[:,0].max() == 0  # assert bs=1
            x = ME.SparseTensor(input, coords, device='cuda')
            x = pool0(x)

            d = {}
            neis_d = x.coordinate_manager.get_kernel_map(x.coordinate_map_key,
                                                                    x.coordinate_map_key,
                                                                    kernel_size=3,
                                                                    stride=1,
                                                                    dilation=1,
                                                                    )
            # d['all_c'] = x.C[:,1:]
            N = x.C.shape[0]
            k = 27
            all_neis = []
            for k_ in range(k):

                if not k_ in neis_d.keys():
                        continue

                neis_ = torch.gather(x.C[:,1:].float(), dim=0, index=neis_d[k_][0].reshape(-1,1).repeat(1,3).long())
                neis = torch.zeros(N,3, device=x.F.device)
                neis.scatter_(dim=0, index=neis_d[k_][1].reshape(-1,1).repeat(1,3).long(), src=neis_)
                neis = (neis.sum(-1)>0).int()
                all_neis.append(neis)
            all_neis = torch.stack(all_neis)
            all_neis_scenes.append(all_neis)

        # d['neis'] = all_neis
        d['sparse_mask'] = torch.cat(all_neis_scenes, dim=-1)
        print('Stride:{} Shape:'.format(stride), d['sparse_mask'].shape)
        # name = "sparse_mask_s{}_scannet.pth".format(stride)
        name = 'test-kitti'
        torch.save(d, '/home/zhaotianchen/project/point-transformer/SpatioTemporalSegmentation-ScanNet/plot/final/{}'.format(name))
        import ipdb; ipdb.set_trace()

    import ipdb; ipdb.set_trace()

