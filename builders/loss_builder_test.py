import tensorflow as tf
from google.protobuf import text_format

from builders.loss_builder import build_loss, build_loss_reduction
from protos import losses_pb2


class LossBuildTest(tf.test.TestCase):
    def test_reduction_build(self):
        self.assertEqual(
            build_loss_reduction(1),
            tf.keras.losses.Reduction.AUTO
        )
        self.assertEqual(
            build_loss_reduction(2),
            tf.keras.losses.Reduction.NONE
        )
        self.assertEqual(
            build_loss_reduction(3),
            tf.keras.losses.Reduction.SUM
        )
        self.assertEqual(
            build_loss_reduction(4),
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )

    def test_binary_cross_entropy(self):
        proto_txt = """
        binary_cross_entropy{
        from_logits : false
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn, tf.keras.losses.BinaryCrossentropy)
        self.assertEqual(loss_fn._fn_kwargs['from_logits'], False)
        self.assertEqual(loss_fn._fn_kwargs['label_smoothing'], 0.0)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)

    def test_categorical_cross_entropy(self):
        proto_txt = """
        categorical_cross_entropy{
            from_logits : true
            label_smoothing: 0.001
            reduction : SUM_OVER_BATCH_SIZE
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn, tf.keras.losses.CategoricalCrossentropy)
        self.assertEqual(loss_fn._fn_kwargs['from_logits'], True)
        self.assertEqual(loss_fn._fn_kwargs['label_smoothing'], 0.001)
        self.assertEqual(loss_fn.reduction,
                         tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def test_categorical_hinge(self):
        proto_txt = """
        categorical_hinge{
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn, tf.keras.losses.CategoricalHinge)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)

    def test_cosine_similarity(self):
        proto_txt = """
        cosine_similarity{
        axis : -1
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn, tf.keras.losses.CosineSimilarity)
        self.assertEqual(loss_fn._fn_kwargs['axis'], -1)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)

    def test_hinge(self):
        proto_txt = """
        hinge{
            reduction : NONE
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn, tf.keras.losses.Hinge)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.NONE)

    def test_huber(self):
        proto_txt = """
        huber{
            delta : 1.1
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn, tf.keras.losses.Huber)
        self.assertEqual(loss_fn._fn_kwargs['delta'], 1.1)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)

    def test_kl_divergence(self):
        proto_txt = """
        kl_divergence{
            
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn, tf.keras.losses.KLDivergence)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)

    def test_log_cosh(self):
        proto_txt = """
        log_cosh{
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn, tf.keras.losses.LogCosh)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)

    def test_mean_absolute_error(self):
        proto_txt = """
        mean_absolute_error{
        reduction : SUM
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn, tf.keras.losses.MeanAbsoluteError)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.SUM)

    def test_mean_absolute_percentage_error(self):
        proto_txt = """
        mean_absolute_percentage_error{
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn,
                              tf.keras.losses.MeanAbsolutePercentageError)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)

    def test_mean_squared_error(self):
        proto_txt = """
        mean_squared_error{
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn,
                              tf.keras.losses.MeanSquaredError)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)

    def test_mean_squared_logarithmic_error(self):
        proto_txt = """
        mean_squared_logarithmic_error{
        reduction : AUTO
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn,
                              tf.keras.losses.MeanSquaredLogarithmicError)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)

    def test_poisson(self):
        proto_txt = """
        poisson{
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn,
                              tf.keras.losses.Poisson)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)

    def test_sparse_categorical_cross_entropy(self):
        proto_txt = """
        sparse_categorical_cross_entropy{
        from_logits: true
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn,
                              tf.keras.losses.SparseCategoricalCrossentropy)
        self.assertEqual(loss_fn._fn_kwargs['from_logits'], True)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)

    def test_squared_hinge(self):
        proto_txt = """
        squared_hinge{
        }
        """
        msg = losses_pb2.Loss()
        text_format.Merge(proto_txt, msg)
        loss_fn = build_loss(msg)
        self.assertIsInstance(loss_fn,
                              tf.keras.losses.SquaredHinge)
        self.assertEqual(loss_fn.reduction, tf.keras.losses.Reduction.AUTO)


if __name__ == "__main__":
    tf.test.main()
