import functools

import tensorflow as tf
from loguru import logger

from builders import augmentations_builder as aug_builder
from protos import preprocessing_pb2


def build_preprocessing(preprocessing_proto):
    image_height = preprocessing_proto.image_height
    image_width = preprocessing_proto.image_width
    resize_protocol_value = preprocessing_proto.resize_protocol
    resize_protocol = build_resize_protocol(resize_protocol_value)
    resize_fn = build_resize_preprocessing_fn(image_height=image_height,
                                              image_width=image_width,
                                              resize_protocol=resize_protocol)
    augmentations_proto = preprocessing_proto.augmentations
    augmentation_fn = aug_builder.build_augmentations(
        augmentations_proto=augmentations_proto)
    return resize_fn, augmentation_fn


def build_resize_protocol(protocol_value):
    protocol_name = preprocessing_pb2._RESIZEPROTOCOL.values_by_number[
        protocol_value].name
    logger.debug('Using Resize method : {}.'.format(protocol_name))
    if protocol_name == 'NEAREST_NEIGHBOR':
        protocol = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    elif protocol_name == 'BILINEAR':
        protocol = tf.image.ResizeMethod.BILINEAR
    elif protocol_name == 'BICUBIC':
        protocol = tf.image.ResizeMethod.BICUBIC
    elif protocol_name == 'GAUSSIAN':
        protocol = tf.image.ResizeMethod.GAUSSIAN
    elif protocol_name == 'LANCZOS3':
        protocol = tf.image.ResizeMethod.LANCZOS3
    elif protocol_name == 'LANCZOS5':
        protocol = tf.image.ResizeMethod.LANCZOS5
    elif protocol_name == 'MITCHELLCUBIC':
        protocol = tf.image.ResizeMethod.MITCHELLCUBIC
    elif protocol_name == 'AREA':
        protocol = tf.image.ResizeMethod.AREA
    else:
        logger.error('Unsupported resize method.')
        raise ValueError('Please refer to the log message above.')
    return protocol


def build_resize_preprocessing_fn(image_height, image_width, resize_protocol):
    if image_height == 0 or image_width == 0:
        logger.debug('No resizing will be done during preprocessing.')
        return None

    resize_fn = functools.partial(
        tf.image.resize,
        size=[image_height, image_width],
        method=resize_protocol
    )

    return resize_fn
