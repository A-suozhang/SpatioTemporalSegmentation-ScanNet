import torch

def check_data(model, train_data_loader, val_data_loader, config):
    train_iter = train_data_loader.__iter__()

    if config.return_transformation:
        coords, input, target, _, _, pointcloud, transformation = data_iter.next()


    import ipdb; ipdb.set_trace()

