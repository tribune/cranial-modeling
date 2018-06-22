import unittest
from cranial.common import logger
from cranial.model_base import StatefulModel
from cranial.online_training import OnlineLearningWrapper, TrainerBase, \
    AccumulatorBase, CountSchedule

log = logger.get('test_online_learning')


class DummyModel(StatefulModel):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.state.n = 0

    def transform(self, record):
        return record * self.state.n

    def update(self, iterable):
        for _ in iterable:
            self.state.n += 1
        return self

    def train(self, iterable):
        return self.update(iterable)


class DummyTrainer(TrainerBase):
    is_ready = True

    def update(self, model, data):
        model.state.n = 1
        return model


class DummyAccum(AccumulatorBase):
    _batch = [1, 2, 3]

    def add(self, record):
        pass

    def get_batch(self):
        return self._batch


class TestModelBase(unittest.TestCase):
    def test_olwrapper_init(self):
        t = DummyTrainer()
        m = DummyModel()
        a = DummyAccum()
        s = CountSchedule(1)
        om = OnlineLearningWrapper(model=m, trainer=t, accumulator=a, schedule=s)
        actual = [
            om.trainer is t,
            om.model is m,
            om.accumulator is a
        ]
        expected = [True, True, True]
        self.assertListEqual(actual, expected, 'should make args as attributes')


if __name__ == '__main__':
    unittest.main()
