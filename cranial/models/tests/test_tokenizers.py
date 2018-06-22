import unittest
import sys
import time
import shutil
import subprocess
sys.path.append('.')    # in case file is run from root dir

from cranial.models.tokenizers import MosesTokenizer


def fetcher():
    return ('this is a test sentence' for _ in range(3))


class TestTokenizers(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree('moses_test')

    @classmethod
    def setUpClass(cls):
        # a list of 3 BOW documents, format: (token_id, count)
        try:
            shutil.rmtree('moses_test')
        except:
            pass
        subprocess.run("git clone https://github.com/moses-smt/mosesdecoder moses_test".split())

    def test_moses_tokenizer(self):
        tk = MosesTokenizer(moses_repo_path='moses_test')
        texts = ['one text.', 'two text?', 'The final text!']
        res = tk.itransform(texts)
        actual = [_ for _ in res]
        expected = ['one text .', 'two text ?', 'The final text !']
        self.assertListEqual(actual, expected, "output should be tokenized")

    def test_time_moses_tokenizer(self):
        tk = MosesTokenizer(moses_repo_path='moses_test')
        with open('cranial/models/tests/data/just_texts.txt') as f:
            texts = f.read().split('\n\n')

        res = tk.itransform(texts * 10)
        t0 = time.time()
        _ = [_ for _ in res]
        print("All separate: ", len(texts) * 10, 'x', sum([len(txt) for txt in texts])// len(texts), '\ttime: ', time.time() - t0)

        texts = ['\n\n'.join(texts)] * 10
        res = tk.itransform(texts)
        t0 = time.time()
        _ = [_ for _ in res]
        print("by file: ", len(texts), 'x', sum([len(txt) for txt in texts])// len(texts), '\ttime: ', time.time() - t0)

        texts = ['\n\n'.join(texts)]
        res = tk.itransform(texts)
        t0 = time.time()
        _ = [_ for _ in res]
        print("All files in one: ", len(texts), 'x', len(texts[0]),'\ttime: ', time.time() - t0)


if __name__ == '__main__':
    unittest.main()