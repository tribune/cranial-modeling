import unittest
import os
from cranial.common import logger
from cranial.model_base import State, StatefulModel, ModelBase


log = logger.get('test_re_iter')


class DummyModel(ModelBase):
    def transform(self, record):
        return record * 2


class DummyStateful(StatefulModel):

    def transform(self, record):
        return record * self.state.n

    def train(self, iterable):
        c = 0
        for _ in iterable:
            c += 1
        self.state.n = c


class TestModelBase(unittest.TestCase):

    def test_State_save(self):
        s = State()
        s.foo = 'bar'
        s.save('tmp_state')
        actual = os.path.isfile('tmp_state')
        os.unlink('tmp_state')
        self.assertTrue(actual, 'should save state into a file')

    def test_State_save_load(self):
        s = State()
        s.foo = 'bar'
        s.save('tmp_state')
        s1 = State.load('tmp_state')
        os.unlink('tmp_state')
        actual = str(s1)
        expected = str(s)
        self.assertEqual(actual, expected, "saved and loaded states should be the same")

    def test_ModelBase_init(self):
        m = DummyModel(foo='bar')
        actual = getattr(m, 'foo', 'not bar')
        expected = 'bar'
        self.assertEqual(actual, expected, 'any argument to init should become an attribute')

    def test_dummyModel(self):
        m = DummyModel()
        inputs = [0, 1, 2, 3, 4]
        out = m.itransform(inputs)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [0, 2, 4, 6, 8] * 2
        self.assertListEqual(actual, expected, 'model should return numbers multiplied by two, twice')

    def test_StatefulModel_init(self):
        m = DummyStateful()
        actual = hasattr(m, 'state')
        self.assertTrue(actual, 'on init model should create a state')

    def test_StatefulModel_init_args(self):
        m = DummyStateful(foo='bar')
        actual = getattr(m, 'foo', 'not bar')
        expected = 'bar'
        self.assertEqual(actual, expected, 'any argument to init should become an attribute')

    def test_StatefulModel_train(self):
        m = DummyStateful()
        inputs = [0, 1, 2, 3, 4]
        m.train(inputs)
        actual = m.state.n
        expected = 5
        self.assertEqual(actual, expected, "should count objects and make result as attribute of state")

    def test_StatefulModel_transform(self):
        m = DummyStateful()
        inputs = [0, 1, 2, 3, 4]
        m.state.n = 2
        out = m.itransform(inputs)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [0, 2, 4, 6, 8] * 2
        self.assertEqual(actual, expected, "model should return numbers multiplied by two, twice")

    def test_StatefulModel_save(self):
        m1 = DummyStateful()
        m1.state.n = 2
        m1.save('tmp_model_state')
        actual = os.path.isfile('tmp_model_state')
        os.unlink('tmp_model_state')
        self.assertTrue(actual, 'should save state to a file')

    def test_StatefulModel_save_load(self):
        m1 = DummyStateful()
        m1.state.n = 2
        m1.save('tmp_model_state')
        m2 = DummyStateful()
        m2.load('tmp_model_state')
        os.unlink('tmp_model_state')
        actual = str(m2.state)
        expected = str(m1.state)
        self.assertEqual(actual, expected, "saved and loaded states should match")

if __name__ == '__main__':
    unittest.main()
