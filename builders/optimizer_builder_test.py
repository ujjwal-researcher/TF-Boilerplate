import tensorflow as tf
from google.protobuf import text_format

from builders.optimizer_builder import build_optimizer
from protos import optimizers_pb2


class TestOptimizerBuild(tf.test.TestCase):
    def test_adadelta_constant_lr(self):
        proto_txt = """
        adadelta{
            learning_schedule{
                constant_learning_rate : 0.001
            }
        }
        """
        msg = optimizers_pb2.Optimizer()
        text_format.Merge(proto_txt, msg)
        opt = build_optimizer(msg)
        self.assertIsInstance(opt, tf.keras.optimizers.Adadelta)
        self.assertEqual(opt._hyper['learning_rate'], 0.001)
        self.assertEqual(opt._hyper['rho'], 0.95)
        self.assertEqual(opt.epsilon, 1E-7)

    def test_adadelta_scheduled_lr(self):
        proto_txt = """
        adadelta{
            learning_schedule{
                inverse_time_decay_schedule{
                    initial_learning_rate: 0.01
                    decay_steps: 10000
                    decay_rate: 0.3
                    staircase : true
                }
            }
            rho : 0.99
        }
        """
        msg = optimizers_pb2.Optimizer()
        text_format.Merge(proto_txt, msg)
        opt = build_optimizer(msg)
        self.assertIsInstance(opt, tf.keras.optimizers.Adadelta)
        self.assertIsInstance(opt._hyper['learning_rate'],
                              tf.keras.optimizers.schedules.InverseTimeDecay)
        self.assertEqual(opt._hyper['rho'], 0.99)
        self.assertEqual(opt.epsilon, 1E-7)

    def test_adagrad(self):
        proto_txt = """
                adagrad{
                    learning_schedule{
                        inverse_time_decay_schedule{
                            initial_learning_rate: 0.01
                            decay_steps: 10000
                            decay_rate: 0.3
                            staircase : true
                        }
                    }
                    epsilon: 1e-8
                }
                """
        msg = optimizers_pb2.Optimizer()
        text_format.Merge(proto_txt, msg)
        opt = build_optimizer(msg)
        self.assertIsInstance(opt, tf.keras.optimizers.Adagrad)
        self.assertIsInstance(opt._hyper['learning_rate'],
                              tf.keras.optimizers.schedules.InverseTimeDecay)
        self.assertEqual(opt._initial_accumulator_value, 0.1)
        self.assertEqual(opt.epsilon, 1E-8)

    def test_adam(self):
        proto_txt = """
        adam{
            learning_schedule{
                inverse_time_decay_schedule{
                            initial_learning_rate: 0.01
                            decay_steps: 10000
                            decay_rate: 0.3
                            staircase : true
                        }
            }
            beta_1 : 0.8
            beta_2 : 0.91
            amsgrad: true
        }
        """
        msg = optimizers_pb2.Optimizer()
        text_format.Merge(proto_txt, msg)
        opt = build_optimizer(msg)
        self.assertIsInstance(opt, tf.keras.optimizers.Adam)
        self.assertIsInstance(opt._hyper['learning_rate'],
                              tf.keras.optimizers.schedules.InverseTimeDecay)
        self.assertEqual(opt._hyper['beta_1'], 0.8)
        self.assertEqual(opt._hyper['beta_2'], 0.91)
        self.assertEqual(opt.amsgrad, True)
        self.assertEqual(opt.epsilon, 1E-7)

    def test_adamax(self):
        proto_txt = """
                adamax{
                    learning_schedule{
                        inverse_time_decay_schedule{
                                    initial_learning_rate: 0.01
                                    decay_steps: 10000
                                    decay_rate: 0.3
                                    staircase : true
                                }
                    }
                    beta_1 : 0.8
                    beta_2 : 0.91
                }
                """
        msg = optimizers_pb2.Optimizer()
        text_format.Merge(proto_txt, msg)
        opt = build_optimizer(msg)
        self.assertIsInstance(opt, tf.keras.optimizers.Adamax)
        self.assertIsInstance(opt._hyper['learning_rate'],
                              tf.keras.optimizers.schedules.InverseTimeDecay)
        self.assertEqual(opt._hyper['beta_1'], 0.8)
        self.assertEqual(opt._hyper['beta_2'], 0.91)
        self.assertEqual(opt.epsilon, 1E-7)

    def test_ftrl(self):
        proto_txt = """
        ftrl{
            learning_schedule{
                inverse_time_decay_schedule{
                                    initial_learning_rate: 0.01
                                    decay_steps: 10000
                                    decay_rate: 0.3
                                    staircase : true
                                }
            }
        }
        """
        msg = optimizers_pb2.Optimizer()
        text_format.Merge(proto_txt, msg)
        opt = build_optimizer(msg)
        self.assertIsInstance(opt, tf.keras.optimizers.Ftrl)
        self.assertIsInstance(opt._hyper['learning_rate'],
                              tf.keras.optimizers.schedules.InverseTimeDecay)
        self.assertEqual(opt._hyper['learning_rate_power'], -0.5)
        self.assertEqual(opt._hyper['l1_regularization_strength'], 0.0)
        self.assertEqual(opt._hyper['l2_regularization_strength'], 0.0)
        self.assertEqual(opt._hyper['beta'], 0.0)
        self.assertEqual(opt._l2_shrinkage_regularization_strength, (0.0))

    def test_nadam(self):
        proto_txt = """
                       nadam{
                           learning_schedule{
                              constant_learning_rate : 0.001
                           }
                           beta_1 : 0.8
                           beta_2 : 0.91
                           epsilon : 0.001
                       }
                       """
        msg = optimizers_pb2.Optimizer()
        text_format.Merge(proto_txt, msg)
        opt = build_optimizer(msg)
        self.assertIsInstance(opt, tf.keras.optimizers.Nadam)
        self.assertEqual(opt._hyper['learning_rate'],
                         0.001)
        self.assertEqual(opt._hyper['beta_1'], 0.8)
        self.assertEqual(opt._hyper['beta_2'], 0.91)
        self.assertEqual(opt.epsilon, 1E-3)

    def test_rmsprop(self):
        proto_txt = """
        rmsprop{
             learning_schedule{
                inverse_time_decay_schedule{
                                    initial_learning_rate: 0.01
                                    decay_steps: 10000
                                    decay_rate: 0.3
                                    staircase : true
                                }
            }
            rho : 0.8
            momentum : 0.1
            centered : true
            epsilon : 1E-1
        }
        """
        msg = optimizers_pb2.Optimizer()
        text_format.Merge(proto_txt, msg)
        opt = build_optimizer(msg)
        self.assertIsInstance(opt, tf.keras.optimizers.RMSprop)
        self.assertIsInstance(opt._hyper['learning_rate'],
                              tf.keras.optimizers.schedules.InverseTimeDecay)
        self.assertEqual(opt._hyper['rho'], 0.8)
        self.assertEqual(opt.epsilon, 1E-1)
        self.assertEqual(opt._hyper['momentum'], 0.1)
        self.assertEqual(opt.centered, True)

    def test_sgd(self):
        proto_txt = """
        sgd{
            learning_schedule{
                inverse_time_decay_schedule{
                                    initial_learning_rate: 0.01
                                    decay_steps: 10000
                                    decay_rate: 0.3
                                    staircase : true
                                }
            }
            momentum : 0.1
            nesterov : false
        }
        """
        msg = optimizers_pb2.Optimizer()
        text_format.Merge(proto_txt, msg)
        opt = build_optimizer(msg)
        self.assertIsInstance(opt, tf.keras.optimizers.SGD)
        self.assertIsInstance(opt._hyper['learning_rate'],
                              tf.keras.optimizers.schedules.InverseTimeDecay)
        self.assertEqual(opt._hyper['momentum'], 0.1)
        self.assertEqual(opt.nesterov, False)


if __name__ == "__main__":
    tf.test.main()
