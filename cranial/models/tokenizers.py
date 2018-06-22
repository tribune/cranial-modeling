"""
tokenizers that do not use spaCy
"""
import subprocess
import os
import collections
import logging

from cranial.common import logger
from cranial.model_base import ModelBase

log = logger.create('tokenizers', os.environ.get('MODELS_LOGLEVEL', logging.WARNING))  # streaming log


class MosesTokenizer(ModelBase):
    name = 'moses_tokenizer'

    def __init__(self, moses_repo_path, language='en', threads=None, **kwargs):
        """
        This wraps around a moses tokenizer - https://github.com/moses-smt/mosesdecoder

        Note that it is much faster to transform few large chunks of text instead of many small ones. So before
        passing strings into this tokenizer it might be good to batch short texts into a
        large one with some known separator between individual texts, and then after split apart again

        Parameters
        ----------
        moses_repo_path
            path to the cloned repo

        language
            language, default 'en', never checked that it can work with others but
            supposedly yes: https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl

        threads
            number of threads to pass to the tokenizer command (did not notice any improvements)

        kwargs
            additional kwargs to pass to the parent class constructor
        """
        super(MosesTokenizer, self).__init__(**kwargs)
        self.moses_repo_path = moses_repo_path
        self.language = language
        self.threads = threads

        self.comm = [os.path.join(self.moses_repo_path, 'scripts/tokenizer/tokenizer.perl'),
                     '-q', '-l', self.language]
        if self.threads is not None and self.threads > 1:
            self.comm += ['-threads', format(self.threads)]

        # check command
        result = subprocess.run(self.comm, input="testing...".encode('utf8'),
                                shell=True, check=False, stderr=subprocess.PIPE)
        if result.returncode:
            raise Exception(result.stderr)
        else:
            log.info("Moses tokenizer command >> " + ' '.join(self.comm) + ' -- OK')

    def transform(self, record: str) -> str:
        """
        transform one record
        Parameters
        ----------
        record
            raw text

        Returns
        -------
            tokenized text (tokens are space-separated)
        """
        result = subprocess.run(self.comm, input=record.encode('utf8'),
                                shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.stdout.decode('utf8').strip()


def add_n_grams(token_list, n=1):
    """
    creates n-grams:
        for n = 2    ['a', 'b', 'c'] -> ['a', 'b', 'c', 'a_b', 'b_c']
    Parameters
    ----------
    token_list
        input list of strings
    n
        n in n-grams, how many adjacent elements to merge
    Returns
    -------
        modified list of strings
    """
    if n < 2:
        return token_list
    else:
        adds = []
        for i in range(2, n + 1):
            adds.extend(['_'.join(tt) for tt in zip(*[token_list[j:] for j in range(i)])])
    return token_list + adds
