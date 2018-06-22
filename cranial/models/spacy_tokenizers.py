"""
tokenizers that use spacy
"""
import spacy

from cranial.common import logger
from cranial.re_iter import ReGenerator
from cranial.model_base import ModelBase
from cranial.models.tokenizers import add_n_grams

log = logger.get(name='tokenizers_spacy', var='MODELS_LOGLEVEL')  # streaming log


class SpacyWrapper(ModelBase):
    name = 'spacy_wrapper'

    def __init__(self, lang='en', in_field=None, out_field=None, batch_size=10000, n_threads=1, **spacy_load_params):
        """
        Use spaCy to transform text records into spacy document objects.

        Parameters
        ----------
        min_length
            min number of characters for a token

        stop_list
            list of tokens to exclude

        n_grams
            add n-grams, if n_grams=2, then 'a b c' -> 'a', 'b', 'c', 'a_b', 'b_c'
        """
        super().__init__(**spacy_load_params)
        self.lang = lang
        self.in_field = in_field
        self.out_field = out_field
        assert (self.in_field is None and self.out_field is None) or \
               (self.in_field is not None and self.out_field is not None)
        self.batch_size = batch_size
        self.n_threads = n_threads
        log.info("loading spacy...")
        self.nlp = spacy.load(lang, **spacy_load_params)

    def transform(self, record: str):
        """
        transform one text into spacy doc

        Parameters
        ----------
        record
            text to transform
        Returns
        -------
            spacy doc
        """
        # spaCy-fy
        return self.nlp(record)

    def itransform(self, iterable, iter_name=None):
        """
        use spacy built-in multiprocessing to transform an iterable of texts
        Parameters
        ----------
        iterable
        iter_name

        Returns
        -------
            generator
        """
        if self.in_field is None:
            texts = iterable
            return ReGenerator(
                lambda:(doc for doc in self.nlp.pipe(texts, batch_size=self.batch_size, n_threads=self.n_threads)))
        else:
            texts = (itm[self.in_field] for itm in iterable)
            return ReGenerator(
                lambda: (
                    {**d, self.out_field: doc}
                    for d, doc in zip(iterable,
                                      self.nlp.pipe(texts, batch_size=self.batch_size, n_threads=self.n_threads))
                )
            )