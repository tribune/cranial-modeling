import unittest
import sys
sys.path.append('.')    # in case file is run from root dir

from cranial.models.spacy_tokenizers import SpacyWrapper
from cranial.re_iter import ReGenerator, ReMap


def fetcher():
    return ('this is a test sentence' for _ in range(3))


class TestTokenizers(unittest.TestCase):

    def test_spacy_wrapper_no_fields(self):
        m = SpacyWrapper()
        x = ReGenerator(fetcher)
        x = m.itransform(x)

        expected = ['this is a test sentence', 'this is a test sentence', 'this is a test sentence']
        actual = [itm.text for itm in x]
        self.assertListEqual(actual, expected, "each item in result should be a spacy doc which has attribute "
                                               "text which should be equal to the original sentence")

    def test_spacy_wrapper_with_fields(self):
        m = SpacyWrapper(in_field='text', out_field='doc')
        x = ReGenerator(fetcher)
        x = ReMap(lambda s: {'text': s}, x)
        x = m.itransform(x)

        expected = ['this is a test sentence', 'this is a test sentence', 'this is a test sentence']
        actual = [itm['doc'].text for itm in x]
        self.assertListEqual(actual, expected, "each item in result should be a dictionary with key'doc' that contains "
                                               "spacy doc which has attribute text which should be equal to "
                                               "the original sentence")

if __name__ == '__main__':
    unittest.main()