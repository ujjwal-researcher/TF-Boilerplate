import tensorflow as tf
from loguru import logger

from protos import losses_pb2


def build_loss(loss_proto):
    loss_type = loss_proto.WhichOneof('loss')
    loss_proto = eval('loss_proto.{}'.format(loss_type))
    if loss_type == 'binary_cross_entropy':
        loss_fn = build_binary_cross_entropy(loss_proto)
    elif loss_type == 'categorical_cross_entropy':
        loss_fn = build_categorical_cross_entropy(loss_proto)
    elif loss_type == 'categorical_hinge':
        loss_fn = build_categorical_hinge(loss_proto)
    elif loss_type == 'cosine_similarity':
        loss_fn = build_cosine_similarity(loss_proto)
    elif loss_type == 'hinge':
        loss_fn = build_hinge(loss_proto)
    elif loss_type == 'huber':
        loss_fn = build_huber(loss_proto)
    elif loss_type == 'kl_divergence':
        loss_fn = build_kl_divergence(loss_proto)
    elif loss_type == 'log_cosh':
        loss_fn = build_los_cosh(loss_proto)
    elif loss_type == 'mean_absolute_error':
        loss_fn = build_mean_absolute_error(loss_proto)
    elif loss_type == 'mean_absolute_percentage_error':
        loss_fn = build_mean_absolute_percentage_error(loss_proto)
    elif loss_type == 'mean_squared_error':
        loss_fn = build_mean_squared_error(loss_proto)
    elif loss_type == 'mean_squared_logarithmic_error':
        loss_fn = build_mean_squared_logarithmic_error(loss_proto)
    elif loss_type == 'poisson':
        loss_fn = build_poisson(loss_proto)
    elif loss_type == 'sparse_categorical_cross_entropy':
        loss_fn = build_sparse_categorical_cross_entropy(loss_proto)
    elif loss_type == 'squared_hinge':
        loss_fn = build_squared_hinge(loss_proto)
    else:
        raise ValueError('A valid loss proto was not provided.')

    return loss_fn


def build_binary_cross_entropy(loss_proto):
    from_logits = loss_proto.from_logits
    label_smoothing = loss_proto.label_smoothing
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building Binary Cross Entropy Loss.')
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        reduction=reduction
    )
    return loss_fn


def build_categorical_cross_entropy(loss_proto):
    from_logits = loss_proto.from_logits
    label_smoothing = loss_proto.label_smoothing
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building Categorical Cross Entropy Loss.')
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        reduction=reduction
    )
    return loss_fn


def build_categorical_hinge(loss_proto):
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building Categorical Hinge Loss.')
    loss_fn = tf.keras.losses.CategoricalHinge(
        reduction=reduction
    )
    return loss_fn


def build_cosine_similarity(loss_proto):
    axis = loss_proto.axis
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building Cosine Similarity Loss.')
    loss_fn = tf.keras.losses.CosineSimilarity(
        axis=axis,
        reduction=reduction
    )
    return loss_fn


def build_hinge(loss_proto):
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building Hinge Loss.')
    loss_fn = tf.keras.losses.Hinge(
        reduction=reduction
    )
    return loss_fn


def build_huber(loss_proto):
    delta = loss_proto.delta
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building Huber Loss.')
    loss_fn = tf.keras.losses.Huber(
        delta=delta,
        reduction=reduction
    )
    return loss_fn


def build_kl_divergence(loss_proto):
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building KLDivergence Loss.')
    loss_fn = tf.keras.losses.KLDivergence(
        reduction=reduction
    )
    return loss_fn


def build_los_cosh(loss_proto):
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building LosCosh Loss.')
    loss_fn = tf.keras.losses.LogCosh(
        reduction=reduction
    )
    return loss_fn


def build_mean_absolute_error(loss_proto):
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building MeanAbsoluteError Loss.')
    loss_fn = tf.keras.losses.MeanAbsoluteError(
        reduction=reduction
    )
    return loss_fn


def build_mean_absolute_percentage_error(loss_proto):
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building MeanAbsolutePercentageError Loss.')
    loss_fn = tf.keras.losses.MeanAbsolutePercentageError(
        reduction=reduction
    )
    return loss_fn


def build_mean_squared_error(loss_proto):
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building MeanSquaredError Loss.')
    loss_fn = tf.keras.losses.MeanSquaredError(
        reduction=reduction
    )
    return loss_fn


def build_mean_squared_logarithmic_error(loss_proto):
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building MeanSquaredLogarithmic Loss.')
    loss_fn = tf.keras.losses.MeanSquaredLogarithmicError(
        reduction=reduction
    )
    return loss_fn


def build_poisson(loss_proto):
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building Poisson Loss.')
    loss_fn = tf.keras.losses.Poisson(
        reduction=reduction
    )
    return loss_fn


def build_sparse_categorical_cross_entropy(loss_proto):
    from_logits = loss_proto.from_logits
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building SparseCategoricalCrossEntropy Loss.')
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=from_logits,
        reduction=reduction
    )
    return loss_fn


def build_squared_hinge(loss_proto):
    reduction = build_loss_reduction(
        loss_proto.reduction
    )
    logger.debug('Building SquaredHinge Loss.')
    loss_fn = tf.keras.losses.SquaredHinge(
        reduction=reduction
    )
    return loss_fn


def build_loss_reduction(reduction_proto):
    reduction_name = losses_pb2._REDUCTION.values_by_number[
        reduction_proto
    ].name
    if reduction_name == 'AUTO':
        return tf.keras.losses.Reduction.AUTO
    elif reduction_name == 'NONE':
        return tf.keras.losses.Reduction.NONE
    elif reduction_name == 'SUM':
        return tf.keras.losses.Reduction.SUM
    elif reduction_name == 'SUM_OVER_BATCH_SIZE':
        return tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    else:
        logger.error('Valid Reduction proto not specified.')
        raise ValueError('Please refer to the above log message.')
