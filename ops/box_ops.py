import tensorflow as tf
from loguru import logger


class BBoxList(object):
    def __init__(self,
                 box_list,
                 label_list,
                 mask_list=None,
                 keypoint_list=None
                 ):
        self._data = dict()
        self._data['boxes'] = box_list
        self._data['labels'] = label_list
        self._data['masks'] = mask_list
        self._data['keypoints'] = keypoint_list

    def add_field(self, name, value):
        if name in self._data.keys():
            logger.error('{} already exists. Cannot overwrite. Use set_field('
                         ').'.format(name))
            raise KeyError('Please see the log message above.')

        self._data[name] = value

    def get_field(self, name):
        if name not in self._data.keys():
            logger.error('{} was not found in the data dictionary.'.format(
                name))
            raise KeyError('Please see the log message above.')

        return self._data[name]

    def set_field(self, name, value):
        self._data[name] = value

    def list_fields(self):
        return list(
            self._data.keys()
        )

    def get_height_width(self):
        boxes = self.get_field('boxes')
        ymin, xmin, ymax, xmax = tf.split(boxes, num_or_size_splits=4, axis=1)
        heights = ymax - ymin
        widths = xmax - xmin
        return heights, widths

    def aspect_ratios(self):
        heights, widths = self.get_height_width()
        aspect_ratios = tf.divide(heights, widths)
        return aspect_ratios

    def get_center_coordinates(self):
        boxes = self.get_field('boxes')
        ymin, xmin, ymax, xmax = tf.split(boxes, num_or_size_splits=4, axis=1)
        heights, widths = self.get_height_width()
        ycenter = ymin + heights / 2
        xcenter = xmin + widths / 2
        return ycenter, xcenter

    def to_absolute_coordinates(self, image_height, image_width):
        boxes = self.get_field('boxes')
        ymin, xmin, ymax, xmax = tf.split(boxes, num_or_size_splits=4, axis=1)
        ymin *= image_height
        ymax *= image_height
        xmin *= image_width
        xmax *= image_width
        return ymin, xmin, ymax, xmax
