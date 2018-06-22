import unittest
import sys

sys.path.append('.')  # in case file is run from root dir

from cranial.models.gensim_models import GensimLDA, GensimLSI, GensimTFIDF, GensimDictionary


class TestGensimModels(unittest.TestCase):
    def setUp(self):
        # a list of 3 BOW documents, format: (token_id, count)
        self.inputs = [
            [
                [0, 2],
                [1, 2],
                [2, 3],
            ],
            [
                [3, 1],
                [4, 4],
            ],
            [
                [4, 1],
                [3, 1],
            ]
        ]
        lda_params = dict(num_topics=2,
                          workers=None,
                          chunksize=2000,
                          passes=50,
                          batch=False,
                          alpha='symmetric',
                          eta=None,
                          decay=0.5,
                          offset=1.0,
                          eval_every=10,
                          iterations=50,
                          gamma_threshold=0.001,
                          random_state=137,
                          minimum_probability=0.01,
                          minimum_phi_value=0.01,
                          per_word_topics=False)
        id2word = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}

        self.lda = GensimLDA(lda_params=lda_params, id2word=id2word)
        self.lda.train(self.inputs)

    def test_gensim_lda_train(self):
        res = [(itm[0][1] > itm[1][1]) * 1 for itm in self.lda.itransform(self.inputs)]
        actual = [res[1] == res[2], res[0] == res[1]]
        expected = [True, False]
        # just the fact that it runs is good enough
        # but first and last document should belong to the same cluster
        self.assertListEqual(actual, expected, "second and third document should belong to the same cluster and "
                                               "different from the first one")

    def test_gensim_lda_topics(self):
        actual = self.lda.state.topic_names
        expected = ['e d', 'c a b d e']
        self.assertListEqual(actual, expected, "should concatenate topic terms")

    def test_gensim_lda_keywords(self):
        actual = self.lda.state.token2topics
        expected = {'a': ['e d', 'c a b d e'],
                    'b': ['e d', 'c a b d e'],
                    'c': ['e d', 'c a b d e'],
                    'd': ['e d', 'c a b d e'],
                    'e': ['e d', 'c a b d e']}
        self.assertDictEqual(actual, expected, "every tocken should be related to every topic...")


if __name__ == '__main__':
    unittest.main()
