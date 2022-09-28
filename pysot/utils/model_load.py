# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch
import torch.nn as nn


logger = logging.getLogger('global')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return missing_keys


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
        map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                        'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    try:
        missing_keys = check_keys(model, pretrained_dict)
    except:
        logger.info('[Warning]: using pretrain as features.\
                Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    try:
        model.load_state_dict(pretrained_dict, strict=False)
    except RuntimeError:
        # most possibly loading a pruned model
        for key, module in model.named_modules():
            # if key is in unused_pretrained_keys, then skip (otherwise key error)
            weight = key + '.weight'
            bias = key + '.bias'
            if (weight or bias) in missing_keys:
                logger.info("Skip adapting module {}".format(key))
                continue
            # torch.nn.BatchNorm2d
            if isinstance(module, nn.BatchNorm2d):
                module.weight = torch.nn.Parameter(pretrained_dict[key + ".weight"])
                module.bias = torch.nn.Parameter(pretrained_dict[key + ".bias"])
                module.num_features = module.weight.size(0)
                module.running_mean = module.running_mean[0 : module.num_features]
                module.running_var = module.running_var[0 : module.num_features]
            # torch.nn.Conv2d
            elif isinstance(module, nn.Conv2d):
                # for conv2d layer, bias and groups should be consider
                module.weight = torch.nn.Parameter(pretrained_dict[key + ".weight"])
                module.out_channels = module.weight.size(0)
                module.in_channels = module.weight.size(1)
                if module.groups is not 1:
                    # group convolution case
                    # only support for MobileNet, pointwise conv
                    module.in_channels = module.weight.size(0)
                    module.groups = module.in_channels
                if key + ".bias" in pretrained_dict:
                    module.bias = torch.nn.Parameter(pretrained_dict[key + ".bias"])
            # torch.nn.Linear
            elif isinstance(module, nn.Linear):
                module.weight = torch.nn.Parameter(pretrained_dict[key + ".weight"])
                if key + ".bias" in pretrained_dict:
                    module.bias = torch.nn.Parameter(pretrained_dict[key + ".bias"])
                module.out_features = module.weight.size(0)
                module.in_features = module.weight.size(1)
    return model


def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']

    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    check_keys(optimizer, ckpt['optimizer'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch
