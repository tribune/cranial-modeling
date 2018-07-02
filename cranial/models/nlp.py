"""
nlp models
"""
import collections
import os
import numpy as np
from collections import Counter

from cranial.common import logger
from cranial.model_base import StatefulModel

log = logger.get(name='nlp_models', var='MODELS_LOGLEVEL')


class BasicDictionary(StatefulModel):
    name = 'basic_dictionary'
    def __init__(self, no_below_raw, no_above_raw, max_num_raw, no_below, no_above, max_num,
                 filter_at=100000, token_is_tuple=False, protected_tokens=None, **kwargs):
        """
        A custom class for creating a dictionary of tokens from given texts

        When creating dictionary from a long list of texts, there is an intermediate filtering of the dictionary,
        the frequency is defined by filter_at number.

        Parameters
        ----------
        no_below_raw
            min token frequency (float 0 to 1 or int number of documents) to keep during intermediate filtering

        no_above_raw
            max token frequency (float 0 to 1) to keep during intermediate filtering

        max_num_raw
            max number of most frequent tokens to keep during intermediate filtering

        no_below
            min token frequency (float between 0 and 1, or an int number of documents it occured in) to be kept in the final dictionary

        no_above
            max token frequency (a float between 0 and 1) to be kept in the final dictionary

        max_num
            max number of tokens in the final dictionary

        filter_at
            frequency of intermediate filtering specified in numbers of training items (docs) passed through

        token_is_tuple
            set True to account for Dandelion annotations where each 'token' in a document is a tuple (token, score)

        protected_tokens
            list of tokens that should always be in the dictionary, essentially added even if
            not encountered in documents

        kwargs
            additional kwargs passed to parent class constructor
        """
        super(BasicDictionary, self).__init__(**kwargs)
        self.no_below_raw = no_below_raw
        self.no_above_raw = no_above_raw
        self.max_num_raw = max_num_raw
        self.no_below = no_below
        self.no_above = no_above
        self.max_num = max_num
        self.filter_at = filter_at
        self.token_is_tuple = token_is_tuple
        self.protected_tokens = protected_tokens

        self.state.frequency = Counter()
        self.state.doc_frequency = Counter()
        self.state.id2token = []
        self.state.token2id = {}
        self.state.size = 0
        self._num_docs = 0

    def __repr__(self):
        return 'BasicDictionary with {} tokens:\t"{}"'.format(
            len(self.state.frequency),
            '", "'.join([itm[0] for itm in collections.Counter(self.state.frequency).most_common(20)])
        )

    def train(self, iterable):
        """
        iterable is a list of lists of tokens

        Parameters
        ----------
        iterable

        Returns
        -------
            self
        """

        for self._num_docs, doc in enumerate(iterable):
            if self.token_is_tuple:
                doc = [t[0] for t in doc]
            if self.protected_tokens is not None:
                doc = [t for t in doc if t not in self.protected_tokens]
            self.state.frequency.update(doc)
            self.state.doc_frequency.update(set(doc))
            if self._num_docs > 0 and self._num_docs % self.filter_at == 0:
                log.info(self.__repr__())
                log.info("filtering at {}".format(self._num_docs))
                self._filter_tokens(self.no_above_raw, self.no_below_raw, self.max_num_raw)

        self._num_docs += 1
        log.info(self.__repr__())
        log.info("Total docs: {}.\tFinal filter:".format(self._num_docs))
        self._filter_tokens(self.no_above, self.no_below, 0)  # will decrease number in the next step

        max_num = self.max_num if self.max_num > 0 else len(self.state.frequency)
        self.state.id2token = [t for t, v in self.state.frequency.most_common(max_num)]
        self.state.frequency = {t: self.state.frequency[t] for t in self.state.id2token}  # fix conters
        self.state.doc_frequency = {t: self.state.doc_frequency[t] for t in self.state.id2token}  # fix conters
        self.state.token2id = {t: i for i, t in enumerate(self.state.id2token)}
        if self.protected_tokens is not None:
            self.state.token2id.update({t: t for t in self.protected_tokens})
        self.state.size = len(self.state.id2token)

        return self

    def _filter_tokens(self, no_above, no_below, max_num):
        """
        helper methods to filter dictionary
        """
        max_freq = int(self._num_docs * no_above)
        if max_freq > 0:
            log.info("filtering for frequency <= {}".format(max_freq))
            bad_tokens = [t for t, v in self.state.doc_frequency.items() if v > max_freq]
            [self.state.doc_frequency.pop(t) for t in bad_tokens]
            [self.state.frequency.pop(t) for t in bad_tokens]

        min_freq = no_below * self._num_docs if no_below < 1 else no_below
        if max_num > 0 and max_num < len(self.state.frequency):
            freqs = sorted(self.state.frequency.values(), reverse=True)
            min_freq = max(freqs[max_num], max_freq)

        if min_freq > 0:
            log.info("filtering for frequency > {}".format(min_freq))
            bad_tokens = [t for t, v in self.state.frequency.items() if v < min_freq]
            [self.state.doc_frequency.pop(t) for t in bad_tokens]
            [self.state.frequency.pop(t) for t in bad_tokens]

    def transform(self, record):
        """
        record is a list of tokens, transform into a list of IDs
        Parameters
        ----------
        record
            list of tokens
        Returns
        -------
            list of IDs
        """
        if len(record) == 0:
            return []

        if self.token_is_tuple:
            return [(self.state.token2id[t], v) for t, v in record if t in self.state.token2id.keys()]

        if isinstance(record[0], str):
            return [self.state.token2id[t] for t in record if t in self.state.token2id.keys()]
        elif isinstance(record[0], (list, tuple, np.ndarray)):
            return [self.transform(sub_list) for sub_list in record]
