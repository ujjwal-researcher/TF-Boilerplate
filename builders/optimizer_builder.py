import tensorflow as tf
from loguru import logger

from builders import learning_schedule_builder as ls_builder


def build_optimizer(optimizer_proto):
    optimizer_type = optimizer_proto.WhichOneof('optimizer')
    optimizer_proto = eval('optimizer_proto.{}'.format(optimizer_type))
    learning_schedule = ls_builder.build_learning_schedule(
        optimizer_proto.learning_schedule
    )
    if optimizer_type == 'adadelta':
        optimizer = build_adadelta(optimizer_proto, learning_schedule)
    elif optimizer_type == 'adagrad':
        optimizer = build_adagrad(optimizer_proto, learning_schedule)

    elif optimizer_type == 'adam':
        optimizer = build_adam(optimizer_proto, learning_schedule)

    elif optimizer_type == 'adamax':
        optimizer = build_adamax(optimizer_proto, learning_schedule)

    elif optimizer_type == 'ftrl':
        optimizer = build_ftrl(optimizer_proto, learning_schedule)

    elif optimizer_type == 'nadam':
        optimizer = build_nadam(optimizer_proto, learning_schedule)

    elif optimizer_type == 'rmsprop':
        optimizer = build_rmsprop(optimizer_proto, learning_schedule)

    elif optimizer_type == 'sgd':
        optimizer = build_sgd(optimizer_proto, learning_schedule)
    else:
        raise ValueError('A valid optimizer proto was not found.')

    return optimizer


def build_adadelta(optimizer_proto, learning_schedule):
    rho = optimizer_proto.rho
    epsilon = optimizer_proto.epsilon
    logger.debug('Building AdaDelta optimizer.')
    optimizer = tf.keras.optimizers.Adadelta(
        learning_rate=learning_schedule,
        rho=rho,
        epsilon=epsilon
    )
    return optimizer


def build_adagrad(optimizer_proto, learning_schedule):
    initial_accumulator_value = optimizer_proto.initial_accumulator_value
    epsilon = optimizer_proto.epsilon
    logger.debug('Building AdaGrad Optimizer.')
    optimizer = tf.keras.optimizers.Adagrad(
        learning_rate=learning_schedule,
        initial_accumulator_value=initial_accumulator_value,
        epsilon=epsilon
    )

    return optimizer


def build_adam(optimizer_proto, learning_schedule):
    beta_1 = optimizer_proto.beta_1
    beta_2 = optimizer_proto.beta_2
    epsilon = optimizer_proto.epsilon
    amsgrad = optimizer_proto.amsgrad
    logger.debug('Building Adam Optimizer.')
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        amsgrad=amsgrad
    )

    return optimizer


def build_adamax(optimizer_proto, learning_schedule):
    beta_1 = optimizer_proto.beta_1
    beta_2 = optimizer_proto.beta_2
    epsilon = optimizer_proto.epsilon
    logger.debug('Building AdaMax Optimizer.')
    optimizer = tf.keras.optimizers.Adamax(
        learning_rate=learning_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon
    )

    return optimizer


def build_ftrl(optimizer_proto, learning_schedule):
    learning_rate_power = optimizer_proto.learning_rate_power
    initial_accumulator_value = optimizer_proto.initial_accumulator_value
    l1_regularization_strength = optimizer_proto.l1_regularization_strength
    l2_regularization_strength = optimizer_proto.l2_regularization_strength
    l2_shrinkage_regularization_strength = optimizer_proto.l2_shrinkage_regularization_strength
    beta = optimizer_proto.beta
    logger.debug('Building Ftrl Optimizer.')
    optimizer = tf.keras.optimizers.Ftrl(
        learning_rate=learning_schedule,
        learning_rate_power=learning_rate_power,
        initial_accumulator_value=initial_accumulator_value,
        l1_regularization_strength=l1_regularization_strength,
        l2_regularization_strength=l2_regularization_strength,
        l2_shrinkage_regularization_strength=l2_shrinkage_regularization_strength,
        beta=beta
    )

    return optimizer


def build_nadam(optimizer_proto, learning_schedule):
    beta_1 = optimizer_proto.beta_1
    beta_2 = optimizer_proto.beta_2
    epsilon = optimizer_proto.epsilon
    logger.debug('Building Nadam Optimizer.')
    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=learning_schedule,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon
    )

    return optimizer


def build_rmsprop(optimizer_proto, learning_schedule):
    rho = optimizer_proto.rho
    momentum = optimizer_proto.momentum
    epsilon = optimizer_proto.epsilon
    centered = optimizer_proto.centered

    logger.debug('Building RMSProp Optimizer.')
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_schedule,
        rho=rho,
        momentum=momentum,
        epsilon=epsilon,
        centered=centered
    )
    return optimizer


def build_sgd(optimizer_proto, learning_schedule):
    momentum = optimizer_proto.momentum
    nesterov = optimizer_proto.nesterov

    logger.debug('Building SGD Optimizer.')

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_schedule,
        momentum=momentum,
        nesterov=nesterov
    )

    return optimizer
