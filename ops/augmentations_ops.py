import tensorflow as tf


def random_grayscale(images, labels=None, boxes=None, masks=None,
                     keypoints=None,
                     gray_probability=0.5):
    if keypoints is not None:
        raise NotImplementedError('Currently keypoints are not supported.')
    do_gray = tf.random.uniform(shape=(), name='do_gray')

    def perform_graying():
        grayed_images = tf.image.rgb_to_grayscale(images=images,
                                                  name='to_grayscale')
        return grayed_images, labels, boxes, masks

    def no_graying():
        with tf.name_scope('No_graying'):
            return images, labels, boxes, masks

    grayed_images, labels, boxes, masks = tf.cond(
        tf.greater(do_gray, 1.0 - gray_probability),
        lambda: perform_graying(),
        lambda: no_graying()
    )

    return grayed_images, labels, boxes, masks


def random_horizontal_flip(images, labels=None, boxes=None, masks=None,
                           keypoints=None,
                           flip_probability=0.5):
    if keypoints is not None:
        raise NotImplementedError('Currently keypoints are not supported.')
    do_flip = tf.random.uniform(shape=(), name='do_flip')

    def perform_flip():
        with tf.name_scope('Horizontal_flip'):
            with tf.name_scope('image'):
                flipped_images = tf.image.flip_left_right(image=images)
            with tf.name_scope('boxes'):
                flipped_boxes = _flip_boxes_left_right(boxes=boxes)
            with tf.name_scope('masks'):
                flipped_masks = _flip_masks_left_right(masks=masks)
        return flipped_images, labels, flipped_boxes, flipped_masks

    def no_flip():
        with tf.name_scope('No_flip'):
            return images, labels, boxes, masks

    flipped_images, labels, flipped_boxes, flipped_masks = tf.cond(
        tf.greater(do_flip, 1.0 - flip_probability, name='test_if_do_flip'),
        lambda: perform_flip(),
        lambda: no_flip()
    )
    return flipped_images, labels, flipped_boxes, flipped_masks


def _flip_boxes_left_right(boxes):
    if boxes is None:
        return None
    ymin, xmin, ymax, xmax = tf.split(value=boxes,
                                      num_or_size_splits=4,
                                      axis=-1
                                      )
    flipped_xmin = tf.subtract(1.0, xmax)
    flipped_xmax = tf.subtract(1.0, xmin)
    flipped_boxes = tf.concat(
        [
            ymin,
            flipped_xmin,
            ymax,
            flipped_xmax
        ],
        axis=-1
    )
    return flipped_boxes


def _flip_masks_left_right(masks):
    if masks is None:
        return None
    return masks[:, :, ::-1]
