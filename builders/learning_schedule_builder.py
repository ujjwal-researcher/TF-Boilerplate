import tensorflow as tf
from loguru import logger


def build_learning_schedule(schedule_proto):
    schedule_type = schedule_proto.WhichOneof('schedule')
    schedule_proto = eval('schedule_proto.{}'.format(schedule_type))
    if schedule_type == 'cosine_decay_schedule':
        schedule = build_cosine_decay_schedule(schedule_proto)
    elif schedule_type == 'cosine_decay_restarts_schedule':
        schedule = build_cosine_decay_restarts_schedule(schedule_proto)
    elif schedule_type == 'exponential_decay_schedule':
        schedule = build_exponential_decay_schedule(schedule_proto)
    elif schedule_type == 'inverse_time_decay_schedule':
        schedule = build_inverse_time_decay_schedule(schedule_proto)
    elif schedule_type == 'piecewise_constant_decay_schedule':
        schedule = build_piecewise_constant_decay_schedule(schedule_proto)
    elif schedule_type == 'polynomial_decay_schedule':
        schedule = build_polynomial_decay_schedule(schedule_proto)
    elif schedule_type == 'constant_learning_rate':
        logger.debug('Building constant learning rate.')
        schedule = schedule_proto
    else:
        raise ValueError('A valid learning schedule proto was not found.')

    return schedule


def build_cosine_decay_schedule(schedule_proto):
    initial_learning_rate = schedule_proto.initial_learning_rate
    decay_steps = schedule_proto.decay_steps
    alpha = schedule_proto.alpha
    logger.debug('Building CosineDecay learning schedule.')
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=alpha
    )
    return schedule


def build_cosine_decay_restarts_schedule(schedule_proto):
    initial_learning_rate = schedule_proto.initial_learning_rate
    first_decay_steps = schedule_proto.first_decay_steps
    t_mul = schedule_proto.t_mul
    m_mul = schedule_proto.m_mul
    alpha = schedule_proto.alpha
    logger.debug('Building CosineDecayRestarts learning schedule.')
    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=first_decay_steps,
        t_mul=t_mul,
        m_mul=m_mul,
        alpha=alpha
    )
    return schedule


def build_exponential_decay_schedule(schedule_proto):
    initial_learning_rate = schedule_proto.initial_learning_rate
    decay_steps = schedule_proto.decay_steps
    decay_rate = schedule_proto.decay_rate
    staircase = schedule_proto.staircase
    logger.debug('Building ExponentialDecay learning schedule.')
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )
    return schedule


def build_inverse_time_decay_schedule(schedule_proto):
    initial_learning_rate = schedule_proto.initial_learning_rate
    decay_steps = schedule_proto.decay_steps
    decay_rate = schedule_proto.decay_rate
    staircase = schedule_proto.staircase
    logger.debug('Building InverseTimeDecay learning schedule.')
    schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )
    return schedule


def build_piecewise_constant_decay_schedule(schedule_proto):
    boundaries = schedule_proto.boundaries
    values = schedule_proto.values
    if len(values) - len(boundaries) - 1 != 0:
        logger.error('Number of elements in values must be 1 more than those '
                     'in boundaries.')
        raise ValueError('Please check the logs.')

    logger.debug('Building PiecewiseConstantDecay learning schedule.')
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries,
        values=values
    )
    return schedule


def build_polynomial_decay_schedule(schedule_proto):
    initial_learning_rate = schedule_proto.initial_learning_rate
    decay_steps = schedule_proto.decay_steps
    end_learning_rate = schedule_proto.end_learning_rate
    power = schedule_proto.power
    cycle = schedule_proto.cycle
    logger.debug('Building PolynomialDecay Schedule.')
    schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        end_learning_rate=end_learning_rate,
        power=power,
        cycle=cycle
    )
    return schedule
