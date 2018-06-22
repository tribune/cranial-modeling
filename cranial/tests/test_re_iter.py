import unittest
import sys

sys.path.append('.')  # in case file is run from root dir
from cranial.re_iter import *
from cranial.common import logger

log = logger.get(name='test_re_iter')


def dummy_fn(x):
    return 2 * x


class TestReIter(unittest.TestCase):

    def test_ReGenerator(self):
        gen_fn = lambda: range(5)
        out = ReGenerator(gen_fn)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [_ for _ in range(5)] + [_ for _ in range(5)]
        self.assertListEqual(actual, expected, 'should repeat 0->4 sequence twice')

    def test_ReFilter(self):
        inpt = [0, 1, 2, 3, 4]
        out = ReFilter(iterable_input=inpt, fn=lambda x: x % 2)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [1, 3, 1, 3]
        self.assertListEqual(actual, expected, 'should leave only odd numbers, twice')

    def test_ReChain(self):
        inpt = [
            [0, 1, 2],
            [3, ],
            [4, 5]
        ]
        out = ReChain(inpt)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = list(range(6)) * 2
        self.assertListEqual(actual, expected, 'should extend into a single sequence, twice')

    def test_ReRepeat(self):
        inpt = [0, 1, 2, ]
        out = ReRepeat(inpt, n=2)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [0, 0, 1, 1, 2, 2] * 2
        self.assertListEqual(actual, expected, 'should repeat each item twice, twice')

    def test_ReCycle(self):
        inpt = [0, 1, 2, ]
        out = ReCycle(inpt, n=2)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [0, 1, 2, ] * 4
        self.assertListEqual(actual, expected, 'should repeat sequence twice, twice')

    def test_ReZip(self):
        inpt_1 = [0, 1, 2, ]
        inpt_2 = [0, 1, 2, 3]
        out = ReZip(inpt_1, inpt_2)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [(0, 0), (1, 1), (2, 2)] * 2
        self.assertListEqual(actual, expected, 'should zip two input sequences (to the end of shortest), twice')

    def test_ReMap_main_proc(self):
        inpt = [0, 1, 2, 3, 4]
        fn = lambda x: 2 * x
        out = ReMap(iterable_input=inpt, fn=fn)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [0, 2, 4, 6, 8] * 2
        self.assertListEqual(actual, expected, 'should apply x2 function to input, twice')

    def test_ReMap_sub_proc(self):
        inpt = [0, 1, 2, 3, 4]
        out = ReMap(iterable_input=inpt, fn=dummy_fn, proc_type='sub', n_proc=2, verbose=True)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [0, 2, 4, 6, 8] * 2
        self.assertListEqual(actual, expected, 'should apply x2 function to input, twice')

    def test_ReMap_threads(self):
        inpt = [0, 1, 2, 3, 4]
        out = ReMap(iterable_input=inpt, fn=dummy_fn, proc_type='th', n_proc=2, verbose=True)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [0, 2, 4, 6, 8] * 2
        self.assertListEqual(actual, expected, 'should apply x2 function to input, twice')

    def test_ReBatch_no_random_only_full(self):
        inpt = [0, 1, 2, 3, 4]
        out = ReBatch(inpt, 2, only_full=True, shuffle=False, buffer_size=None)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [[0, 1], [2, 3]] * 2
        self.assertListEqual(actual, expected, 'should make two batches only, twice')

    def test_ReBatch_no_random(self):
        inpt = [0, 1, 2, 3, 4]
        out = ReBatch(inpt, 2, only_full=False, shuffle=False, buffer_size=None)
        actual = [_ for _ in out] + [_ for _ in out]
        expected = [[0, 1], [2, 3], [4]] * 2
        self.assertListEqual(actual, expected, 'should make two batches only, twice')

    def test_ReBatch_random_only_full_len(self):
        inpt = [0, 1, 2, 3, 4]
        out = ReBatch(inpt, 2, only_full=True, shuffle=True, buffer_size=None)
        res = [_ for _ in out] + [_ for _ in out]
        actual = [
            len(res),
            len(res[0]),
            sum([len(r) for r in res])
        ]
        expected = [4, 2, 8]
        self.assertListEqual(actual, expected, 'should produce 4 lists of 2 items each')

    def test_ReBatch_no_random_len(self):
        inpt = [0, 1, 2, 3, 4]
        out = ReBatch(inpt, 2, only_full=False, shuffle=True, buffer_size=None)
        res = [_ for _ in out] + [_ for _ in out]
        actual = [
            len(res),
            len(res[0]),
            len(res[2]),
            sum([len(r) for r in res])
        ]
        expected = [6, 2, 1, 10]
        self.assertListEqual(actual, expected, 'should produce two lists of two items, then '
                                               'list of one item, then repeat')

    def test_ReBatch_random_full_only_2iters_difference(self):
        inpt = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # larger sequence to decrease the chance of accidental order
        out = ReBatch(inpt, 4, only_full=True, shuffle=True, buffer_size=None)
        actual_0 = [_ for _ in out]
        actual_1 = [_ for _ in out]
        self.assertNotEqual(actual_0, actual_1, 'should randomize every iteration')

    def test_ReBatch_random_full_only(self):
        inpt = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # larger sequence to decrease the chance of accidental order
        out = ReBatch(inpt, 4, only_full=True, shuffle=True, buffer_size=None)
        actual_0 = [_ for _ in out]
        actual_1 = [_ for _ in out]
        actual = actual_0 + actual_1
        notexpected = [[0, 1, 2, 3], [4, 5, 6, 7]] * 2
        self.assertNotEqual(actual, notexpected, 'should not return items in order')

    def test_BucketBatch_no_random_only_full(self):
        inpt = [
            [0, 1, 2],
            [0],
            [0, 1],
            [3, 4, 5, 6, 7],
            [1],
            [2, 3],
            [7, 8, 9],
            [2],
            [4, 5],
        ]
        out = BucketBatch(inpt, batch_size=2, buckets=[0, 1, 2, 4], pad_index=-1,
                          only_full=True, shuffle=False, buffer_size=None)
        actual = [_ for _ in out]

        expected = [
            [
                [0, 1, 2, -1],
                [3, 4, 5, 6],
            ],
            [
                [0],
                [1],
            ],
            [
                [0, 1],
                [2, 3]
            ]
        ]
        self.assertListEqual(actual, expected, 'first should return a list of two lists of 4 items, '
                                               'first list should end with padding index: -1; then '
                                               'should return list of two lists of one item each; finally '
                                               'should return list of two lists of 2 items each')

    def test_ReBucketBatch_no_random(self):
        inpt = [
            [0, 1, 2],
            [0],
            [0, 1],
            [3, 4, 5, 6, 7],
            [1],
            [2, 3],
            [7, 8, 9],
            [2],
            [4, 5],
        ]
        out = BucketBatch(inpt, batch_size=2, buckets=[0, 1, 2, 4], pad_index=-1,
                          only_full=False, shuffle=False, buffer_size=None)
        actual = [_ for _ in out]

        expected = [
            [
                [0, 1, 2, -1],
                [3, 4, 5, 6],
            ],
            [
                [0],
                [1],
            ],
            [
                [0, 1],
                [2, 3]
            ],
            [
                [2],
            ],
            [
                [4, 5],
            ],
            [
                [7, 8, 9, -1]
            ]
        ]
        self.maxDiff = None
        self.assertListEqual(actual, expected, 'first should return a list of two lists of 4 items, '
                                               'first list should end with padding index: -1; then '
                                               'should return list of two lists of one item each; finally '
                                               'should return list of two lists of 2 items each; after that '
                                               'start returning lists of one list of increasing lengths: 1, 2, 4; '
                                               'final list should have last item -1 again')

    def test_BucketBatch_random_only_full_len(self):
        inpt = [
            [0, 1, 2],
            [0],
            [0, 1],
            [3, 4, 5, 6],
            [1],
            [2, 3],
            [7, 8, 9],
            [2],
            [4, 5],
        ]
        out = BucketBatch(inpt, batch_size=2, buckets=[0, 1, 2, 4], pad_index=-1,
                          only_full=True, shuffle=True, buffer_size=10)
        res = [_ for _ in out]
        actual = [
            [len(r) for r in res],
            [len(r[0]) for r in res],
        ]
        expected = [
            [2, 2, 2],
            [1, 2, 4]
        ]
        self.assertListEqual(actual, expected, 'since buffer is large, it waits till the end and '
                                               'then returns in order of increasing length')

    def test_BucketBatch_randomization_only_full(self):
        inpt = [
            [0], [1],
            [2], [3],
            [4], [5],
            [6], [7],
            [8], [9],
        ]
        out = BucketBatch(inpt, batch_size=2, buckets=[0, 1, ], pad_index=-1,
                          only_full=True, shuffle=True, buffer_size=10)
        actual = [_ for _ in out]
        notexpected = [
            [[0], [1]],
            [[2], [3]],
            [[4], [5]],
            [[6], [7]],
            [[8], [9]],
        ]
        self.assertNotEqual(actual, notexpected, 'sequence should be randomized')

    def test_DiskCache(self):
        inpt = [0, 1, 2, 3, 4]
        out = DiskCache(inpt, tmp_file_path='tmp_test_file')
        file_path = out.tmp_filename
        res1 = [_ for _ in out]
        actual = [os.path.isfile(file_path)]
        res2 = [_ for _ in out]
        del out
        actual.append(os.path.isfile(file_path))
        actual.append(res1)
        actual.append(res2)
        expected = [True, False, inpt, inpt]
        self.assertListEqual(actual, expected, "should make file, then delete at the end, in "
                                               "the meantime should produce input seq twice")


if __name__ == '__main__':
    unittest.main()
