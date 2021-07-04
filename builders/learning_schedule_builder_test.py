import tensorflow as tf
from google.protobuf import text_format

from builders.learning_schedule_builder import build_learning_schedule
from protos import learning_schedules_pb2


class LearningScheduleBuildTest(tf.test.TestCase):
    def test_cosine_decay_schedule(self):
        proto_txt = """
        cosine_decay_schedule{
            initial_learning_rate : 0.001,
            decay_steps : 10000
            alpha: 0.0
        }
        """
        msg = learning_schedules_pb2.LearningRateSchedule()
        text_format.Merge(proto_txt, msg)
        schedule = build_learning_schedule(msg)
        self.assertIsInstance(schedule,
                              tf.keras.optimizers.schedules.CosineDecay)
        self.assertAlmostEqual(schedule.initial_learning_rate, 0.001)
        self.assertAlmostEqual(schedule.alpha, 0.0)
        self.assertAlmostEqual(schedule.decay_steps, 10000)

    def test_cosine_decay_restarts_schedule(self):
        proto_txt = """
        cosine_decay_restarts_schedule{
            initial_learning_rate: 0.0001
            first_decay_steps : 34
            t_mul: 1.0
            m_mul : 0.02
            alpha : 2.0
        }
        """
        msg = learning_schedules_pb2.LearningRateSchedule()
        text_format.Merge(proto_txt, msg)
        schedule = build_learning_schedule(msg)
        self.assertIsInstance(schedule,
                              tf.keras.optimizers.schedules.CosineDecayRestarts)
        self.assertAlmostEqual(schedule.initial_learning_rate, 0.0001)
        self.assertAlmostEqual(schedule.first_decay_steps, 34)
        self.assertAlmostEqual(schedule._t_mul, 1.0)
        self.assertAlmostEqual(schedule._m_mul, 0.02)
        self.assertAlmostEqual(schedule.alpha, 2.0)

    def test_exponential_decay_schedule(self):
        proto_txt = """
        exponential_decay_schedule{
            initial_learning_rate : 0.01
            decay_steps : 10000
            decay_rate : 0.3
            staircase : true
        }
        """
        msg = learning_schedules_pb2.LearningRateSchedule()
        text_format.Merge(proto_txt, msg)
        schedule = build_learning_schedule(msg)
        self.assertIsInstance(schedule,
                              tf.keras.optimizers.schedules.ExponentialDecay)
        self.assertAlmostEqual(schedule.initial_learning_rate, 0.01)
        self.assertAlmostEqual(schedule.decay_steps, 10000)
        self.assertAlmostEqual(schedule.decay_rate, 0.3)
        self.assertAlmostEqual(schedule.staircase, True)

    def test_inverse_time_decay_schedule(self):
        proto_txt = """
                inverse_time_decay_schedule{
                    initial_learning_rate : 0.01
                    decay_steps : 10000
                    decay_rate : 0.3
                    staircase : true
                }
                """
        msg = learning_schedules_pb2.LearningRateSchedule()
        text_format.Merge(proto_txt, msg)
        schedule = build_learning_schedule(msg)
        self.assertIsInstance(schedule,
                              tf.keras.optimizers.schedules.InverseTimeDecay)
        self.assertAlmostEqual(schedule.initial_learning_rate, 0.01)
        self.assertAlmostEqual(schedule.decay_steps, 10000)
        self.assertAlmostEqual(schedule.decay_rate, 0.3)
        self.assertAlmostEqual(schedule.staircase, True)

    def test_piecewise_constant_decay_schedule_valid(self):
        proto_txt = """
        piecewise_constant_decay_schedule{
            boundaries:10000
            boundaries: 15000
            boundaries : 20000
            values : 0.1
            values : 0.01
            values : 0.0001
            values : 0.00001
        }
        """
        msg = learning_schedules_pb2.LearningRateSchedule()
        text_format.Merge(proto_txt, msg)
        schedule = build_learning_schedule(msg)
        self.assertIsInstance(schedule,
                              tf.keras.optimizers.schedules.PiecewiseConstantDecay)
        self.assertAlmostEqual(schedule.boundaries, [10000, 15000, 20000])
        self.assertAlmostEqual(schedule.values, [0.1, 0.01, 0.0001, 0.00001])

    def test_polynomial_decay_schedule(self):
        proto_txt = """
        polynomial_decay_schedule{
            initial_learning_rate : 0.01
            decay_steps : 10000
            end_learning_rate : 0.00001
            power : 0.15
            cycle : true
        }
        """
        msg = learning_schedules_pb2.LearningRateSchedule()
        text_format.Merge(proto_txt, msg)
        schedule = build_learning_schedule(msg)
        self.assertIsInstance(schedule,
                              tf.keras.optimizers.schedules.PolynomialDecay)
        self.assertAlmostEqual(schedule.initial_learning_rate, 0.01)
        self.assertAlmostEqual(schedule.decay_steps, 10000)
        self.assertAlmostEqual(schedule.end_learning_rate, 0.00001)
        self.assertAlmostEqual(schedule.power, 0.15)
        self.assertAlmostEqual(schedule.cycle, True)


if __name__ == "__main__":
    tf.test.main()
