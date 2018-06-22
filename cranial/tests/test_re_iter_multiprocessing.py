import sys
sys.path.append('.')    # in case file is run from root dir
import unittest
import numpy as np
from cranial.common import logger
from cranial.re_iter_multiprocessing import *


log = logger.get('test_re_iter')


def dummy_fn(x):
    return 2 * x


class TestReIter(unittest.TestCase):

    def test_just_an_example(self):
        x = list('kjhiwuhewkaalskjqoik c2837y2hdins')

        i2q = IterToQueue(x)

        step = Step(MapOperator, operator_kwargs={'fn': ord}, previous_step=i2q, n_workers=2)

        step2 = Step(MapOperator, operator_kwargs={'fn': chr}, previous_step=step, n_workers=2)

        step3 = Step(BatchOp,
                     operator_kwargs={'batch_size': 4, 'buffer_size': 100, 'only_full': False, 'shuffle': True},
                     previous_step=step2, n_workers=2)

        res = FromQueue(step3)

        actual1 = [r for r in res]
        actual2 = [r for r in res]
        log.info("Example using multiprocessing steps:\nOnce:\n" + str(actual1) + '\nTwice:\n' + str(actual2))

    def test_IterToQueue(self):
        inpt = [0, 1, 2, 3, 4]
        i2q = IterToQueue(inpt)
        i2q.start()
        actual = []
        while True:
            obj = i2q.out_q.get()
            if obj is None:
                break
            actual.append(obj)
        expected = [0, 1, 2, 3, 4]
        self.assertListEqual(actual, expected, 'should get out what put in')

    def test_FromQueue(self):
        inpt = [0, 1, 2, 3, 4]
        i2q = IterToQueue(inpt)
        out = FromQueue(i2q)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [0, 1, 2, 3, 4] * 2
        self.assertListEqual(actual, expected, 'should get out what put in, twice')

    def test_MapOperator_1_proc(self):
        inpt = [0, 1, 2, 3, 4]
        in_q = Queue()
        out_q = Queue()
        in_q.name = 'in queue'
        out_q.name = 'out queue'
        map_op = MapOperator(dummy_fn, in_q, out_q, n_siblings=1)
        map_op.start()
        for i in inpt:
            in_q.put(i)
        in_q.put(None)
        actual = []
        while True:
            obj = out_q.get()
            if obj is None:
                break
            actual.append(obj)
        expected = [0, 2, 4, 6, 8]
        self.assertListEqual(actual, expected, 'should multiply each input by 2')

    def test_MapOperator_2_proc(self):
        inpt = [0, 1, 2, 3, 4]
        in_q = Queue()
        out_q = Queue()
        in_q.name = 'in queue'
        out_q.name = 'out queue'
        map_ops = [MapOperator(dummy_fn, in_q, out_q, n_siblings=2, verbose=False) for _ in range(2)]
        [op.start() for op in map_ops]
        for i in inpt:
            in_q.put(i)
        in_q.put(None)
        actual = []
        while True:
            obj = out_q.get()
            if obj is None:
                break
            actual.append(obj)
        actual.sort() # since now they can mix order
        expected = [0, 2, 4, 6, 8]
        self.assertListEqual(actual, expected, 'should multiply each input by 2')

    def test_MapOperator_1_proc_seq(self):
        inpt = [0, 1, 2, 3, 4]
        in_q = Queue()
        out_q = Queue()
        in_q.name = 'in queue'
        out_q.name = 'out queue'
        map_op = MapOperator(lambda i: 'a' * i, in_q, out_q, n_siblings=1, res_is_sequence=True)
        map_op.start()
        for i in inpt:
            in_q.put(i)
        in_q.put(None)
        actual = []
        while True:
            obj = out_q.get()
            if obj is None:
                break
            actual.append(obj)
        expected = ['a'] * 10
        self.assertListEqual(actual, expected, 'should return single list of (0+1+2+3+4)=10 letters "a"')

    def test_Step(self):
        inpt = [0, 1, 2, 3, 4]
        i2q = IterToQueue(inpt)
        step = Step(MapOperator,
                    operator_kwargs={'fn': dummy_fn, 'res_is_sequence': False},
                    previous_step=i2q,
                    n_workers=2)
        out = FromQueue(step)
        actual = [_ for _ in out] + [_ for _ in out]
        actual.sort()       # since order is not guaranteed with n_workers > 1
        expected = [0, 0, 2, 2, 4, 4, 6, 6, 8, 8]
        self.assertListEqual(actual, expected, 'should multiply each input by 2, twice')

    def test_BatchOp_no_random_with_last(self):
        inpt = [0, 1, 2, 3, 4]
        i2q = IterToQueue(inpt)
        batch = Step(BatchOp,
                    operator_kwargs={'batch_size': 2, 'shuffle': False, 'only_full': False, },
                    previous_step=i2q,
                    n_workers=1)
        out = FromQueue(batch)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [[0, 1], [2, 3], [4], [0, 1], [2, 3], [4]]
        self.assertListEqual(actual, expected, 'should make batches of two, and then last batch of size 1, twice')

    def test_BatchOp_no_random_full_only(self):
        inpt = [0, 1, 2, 3, 4]
        i2q = IterToQueue(inpt)
        batch = Step(BatchOp,
                    operator_kwargs={'batch_size': 2, 'shuffle': False, 'only_full': True, },
                    previous_step=i2q,
                    n_workers=1)
        out = FromQueue(batch)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [[0, 1], [2, 3], [0, 1], [2, 3]]
        self.assertListEqual(actual, expected, 'should make only batches of two, twice')

    def test_BatchOp_random_full_only(self):
        inpt = [0, 1, 2, 3, 4]
        i2q = IterToQueue(inpt)
        batch = Step(BatchOp,
                    operator_kwargs={'batch_size': 2, 'shuffle': True, 'only_full': True, 'buffer_size': 3},
                    previous_step=i2q,
                    n_workers=1)
        out = FromQueue(batch)
        actual = [_ for _ in out] + [_ for _ in out]
        notexpected = [[0, 1], [2, 3], [0, 1], [2, 3]]
        self.assertNotEqual(actual, notexpected, 'should make random batches of two only, twice')

    def test_BatchOp_random_full_only_len_1(self):
        inpt = [0, 1, 2, 3, 4]
        i2q = IterToQueue(inpt)
        batch = Step(BatchOp,
                    operator_kwargs={'batch_size': 2, 'shuffle': True, 'only_full': True, 'buffer_size': 3},
                    previous_step=i2q,
                    n_workers=1)
        out = FromQueue(batch)
        res = [_ for _ in out] + [_ for _ in out]
        actual = [len(res), [len(r) for r in res]]
        expected = [4, [2, 2, 2, 2]]
        self.assertListEqual(actual, expected, 'should make 4 batches of length 2')

    def test_BatchOp_random_full_only_len_2(self):
        inpt = [0, 1, 2, 3, 4]
        i2q = IterToQueue(inpt)
        batch = Step(BatchOp,
                    operator_kwargs={'batch_size': 2, 'shuffle': True, 'only_full': True, 'buffer_size': 5},
                    previous_step=i2q,
                    n_workers=1)
        out = FromQueue(batch)
        res = [_ for _ in out] + [_ for _ in out]
        actual = [len(res), [len(r) for r in res]]
        expected = [4, [2, 2, 2, 2]]
        self.assertListEqual(actual, expected, 'should make 4 batches of length 2')

    def test_BatchOp_random_full_only_len_3(self):
        inpt = [0, 1, 2, 3, 4]
        i2q = IterToQueue(inpt)
        batch = Step(BatchOp,
                    operator_kwargs={'batch_size': 2, 'shuffle': True, 'only_full': True, 'buffer_size': 50},
                    previous_step=i2q,
                    n_workers=1)
        out = FromQueue(batch)
        res = [_ for _ in out] + [_ for _ in out]
        actual = [len(res), [len(r) for r in res]]
        expected = [4, [2, 2, 2, 2]]
        self.assertListEqual(actual, expected, 'should make 4 batches of length 2')

if __name__ == '__main__':
    unittest.main()
