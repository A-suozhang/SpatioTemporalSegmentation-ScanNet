# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import models.resnet as resnet
import models.res16unet as res16unet
import models.mink_transformer as mink_transformer
import models.point_transformer as point_transformer

MODELS = []


def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if ('Net' in a or 'Transformer' in a)])


add_models(resnet)
add_models(res16unet)
add_models(mink_transformer)
add_models(point_transformer)


def get_models():
  '''Returns a tuple of sample models.'''
  return MODELS


def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_models = get_models()
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    print('Invalid model index. Options are:')
    # Display a list of valid model names
    for model in all_models:
      print('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]

  return NetClass
