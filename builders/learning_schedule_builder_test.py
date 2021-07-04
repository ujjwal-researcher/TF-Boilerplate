import tensorflow as tf
from protos import learning_schedules_pb2
from google.protobuf import text_format
from builders.learning_schedule_builder import build_learning_schedule


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
        self.assertIsInstance(schedule, tf.keras.optimizers.schedules.CosineDecay)
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
        self.assertIsInstance(schedule, tf.keras.optimizers.schedules.CosineDecayRestarts)
        self.assertAlmostEqual(schedule.initial_learning_rate, 0.0001)
        self.assertAlmostEqual(schedule.first_decay_steps, 34)
        self.assertAlmostEqual(schedule._t_mul, 1.0)
        self.assertAlmostEqual(schedule._m_mul, 0.02)
        self.assertAlmostEqual(schedule.alpha, 2.0)


if __name__ == "__main__":
    tf.test.main()
