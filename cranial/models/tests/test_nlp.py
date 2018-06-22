import unittest
import sys
sys.path.append('.')    # in case file is run from root dir
from cranial.models.nlp import BasicDictionary


def fetcher():
    return ('this is a test sentence' for _ in range(3))


class TestNLPModels(unittest.TestCase):

    def test_BasicDictionary_train_no_filter(self):
        inputs = [
            ['b', 'b'],
            ['b', 'c', 'd', 'c', 'd'],
            ['a', 'b', 'c']
        ]
        d = BasicDictionary(no_below_raw=0, no_above_raw=1, max_num_raw=10,
                            no_below=0, no_above=1, max_num=10,
                            filter_at=10, token_is_tuple=False, protected_tokens=None)
        d.train(inputs)
        actual = {attr: getattr(d.state, attr, None)
                  for attr in ['doc_frequency', 'frequency', 'id2token', 'token2id', 'size']}
        expected = {
            'doc_frequency': {'b': 3, 'c': 2, 'd': 1, 'a': 1},
            'frequency': {'b': 4, 'c': 3, 'd': 2, 'a': 1},
            'id2token': ['b', 'c', 'd', 'a'],
            'token2id': {'b': 0, 'c': 1, 'd': 2, 'a': 3},
            'size': 4
        }
        self.assertDictEqual(actual, expected, "should count document and overall frequencies; "
                                               "create id<->token maps; "
                                               "count vocab size")

    # TODO: make more tests for dictionary


if __name__ == '__main__':
    unittest.main()