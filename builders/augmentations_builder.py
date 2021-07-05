import functools

from loguru import logger

import ops.augmentations_ops as aug_ops


def build_augmentations(augmentations_proto):
    augmentations = augmentations_proto.augment_method
    augmentation_fns = list()
    for augmentation in augmentations:
        augmentation_type = augmentation.WhichOneof('augmentation')
        augmentation = eval('augmentation.{}'.format(augmentation_type))
        if augmentation_type == 'random_horizontal_flip':
            fn = build_random_horizontal_flip(augmentation)
        elif augmentation_type == 'random_gray_scale':
            fn = build_random_grayscale(augmentation)
        else:
            logger.error('Unsupported augmentation provided.')
            raise ValueError('Please see the log message above.')

        augmentation_fns.append(fn)
    augmentation_fn = lambda x: functools.reduce(
        lambda acc, x: x(*acc), augmentation_fns, x
    )
    return augmentation_fn


def build_random_horizontal_flip(augmentation_proto):
    logger.debug('Building Random Horizontal flip.')
    flip_probability = augmentation_proto.flip_probability
    flip_fn = functools.partial(
        aug_ops.random_horizontal_flip,
        flip_probability=flip_probability
    )
    return flip_fn


def build_random_grayscale(augmentation_proto):
    logger.debug('Building Random Grayscale.')
    gray_probability = augmentation_proto.gray_probability
    flip_fn = functools.partial(
        aug_ops.random_grayscale,
        gray_probability=gray_probability
    )
    return flip_fn
