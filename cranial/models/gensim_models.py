"""
This file has primitive models that wrap around gensim common models such as LSI, TFIDF, etc...
"""
import gensim as g
import os
from cranial.re_iter import ReMap, DiskCache
from cranial.model_base import StatefulModel, ModelBase
from cranial.common import logger

log = logger.get(name='gensim_models', var='MODELS_LOGLEVEL')  # streaming log


class GensimDictionary(StatefulModel):
    name = 'gensim_dictionary'
    def __init__(self, dict_params: dict, **kwargs):
        """
        Wraps around gensim's Similarity index

        Parameters
        ----------
        sim_params
            kwargs to pass to gensim's Similarity initialization
            This must have `output_prefix` and `num_features`

        kwargs
            any other kwargs to be passed to parent class __init__
        """
        super(GensimDictionary, self).__init__(**kwargs)
        self.params = dict_params
        self.state.model = None
        log.info("Init gensim dictionary with params:\n{}".format(dict_params))

    def transform(self, record):
        """
        Dictionary transforms list of tokens into list bow document
        Parameters
        ----------
        record
            list of tokens

        Returns
        -------
            BoW document: list of tuples (token_id, count)
        """
        return self.state.model.doc2bow(record)

    def train(self, iterable):
        """
        each item in iterable is a list of tokens
        """
        log.info("Building gensim dictionary...")
        self.state.model = g.corpora.Dictionary()

        batch = []
        for i, doc in enumerate(iterable):
            batch.append(doc)
            # occasionally dump into dictionary,
            if i > 0 and i % (self.params['dict_filter_every'] // 5) == 0:
                self.state.model.add_documents(batch)
                batch = []

            # occasionally filter
            if i > 0 and i % self.params['dict_filter_every'] == 0:
                log.info("Current dictionary: {}".format(self.state.model))
                log.info("Filtering at {} documents".format(i))
                self.state.model.filter_extremes(no_below=self.params['no_below_raw'],
                                                 no_above=self.params['no_above_raw'],
                                                 keep_n=self.params['max_n_raw'])
                self.state.model.compactify()
                log.info("Now dictionary: {}".format(self.state.model))

        # finalize
        self.state.model.add_documents(batch)
        self.state.model.filter_extremes(no_below=self.params['no_below_raw'],
                                         no_above=self.params['no_above_raw'],
                                         keep_n=self.params['max_n_raw'])
        self.state.model.compactify()
        log.info("Final raw dictionary: {}".format(self.state.model))

        self._reduce_dictionary()
        self.state.model.id2token = {v: k for k, v in self.state.model.token2id.items()}
        return self

    def _reduce_dictionary(self):
        '''
        # make a smaller version and also save
        Parameters
        ----------
        dict_filter

        Returns
        -------

        '''
        # optionally remove certain tokens
        if self.params.get('bad_tokens') is not None:
            bad_ids = [self.state.model.token2id[t]
                       for t in self.params.get('bad_tokens')
                       if t in self.state.model.token2id.keys()]
            self.state.model.filter_tokens(bad_ids=bad_ids)

        # apply new filter
        self.state.model.filter_extremes(no_below=self.params['no_below'],
                                         no_above=self.params['no_above'],
                                         keep_n=self.params['max_n'])
        self.state.model.compactify()
        log.info("Final dictionary: {}".format(self.state.model))


class GensimSimilarity(StatefulModel):
    name = 'gensim_similarity'
    def __init__(self, sim_params: dict, **kwargs):
        """
        Wraps around gensim's Similarity index

        Parameters
        ----------
        sim_params
            kwargs to pass to gensim's Similarity initialization
            This must have `output_prefix` and `num_features`

        kwargs
            any other kwargs to be passed to parent class __init__
        """
        super(GensimSimilarity, self).__init__(**kwargs)
        self.params = sim_params
        self.state.model = None
        self.state.doc_index = []

    def transform(self, record):
        """
        Record is a gensim doc - list of tuples (dim_id, score)
        """
        if self.state.doc_index:
            return [(self.state.doc_index[ix], float(sc)) for ix, sc in self.state.model[record]]
        else:
            return [(int(ix), float(sc)) for ix, sc in self.state.model[record]]

    def train(self, iterable):
        """
        Either each item is a tuple (some_str_ID, doc), or just a doc,
        where doc is a list of tuples (dim_id, score)
        """
        iterable = DiskCache(iterable)
        self.state.doc_index = [itm[0] for itm in iterable]
        corpus = ReMap(iterable, lambda itm: itm[1])
        self.state.model = g.similarities.Similarity(corpus=corpus, **self.params)
        return self

    def update(self, iterable):
        """
        Either each item is a tuple (some_str_ID, doc), or just a doc,
        where doc is a list of tuples (dim_id, score)
        """
        iterable = DiskCache(iterable)
        self.state.doc_index.extend([itm[0] for itm in iterable])
        corpus = ReMap(iterable, lambda itm: itm[1])
        self.state.model.add_documents(corpus)
        return self


class GensimLSI(StatefulModel):
    name = 'gensim_lsi'
    def __init__(self, lsi_params: dict, id2word: dict = None, **kwargs):
        """
        Wraps around gensim's LDA model

        Parameters
        ----------
        lsi_params
            kwargs to pass to gensim's LSI model initialization

        id2word
            id2word to pass to gensim's LSI model initialization, separate from
            lda_params because id2word needs to be obtained by training a dictionary

        kwargs
            any other kwargs to be passed to parent class __init__
        """
        super(GensimLSI, self).__init__(**kwargs)
        self.params = lsi_params
        self.id2word = id2word
        self.state.model = None
        self.state.topic_names = []
        log.info("Init gensim LSI with params:\n{}".format(self.params))

    def transform(self, record):
        return [(int(ix), float(sc)) for ix, sc in self.state.model[record]]

    def train(self, iterable):
        self.state.model = g.models.LsiModel(corpus=iterable, id2word=self.id2word, **self.params)

        # need topic names
        self.state.topic_names = []
        for i in range(self.state.model.num_topics):
            vals = self.state.model.show_topic(i, 100)
            v0 = abs(vals[0][1])
            vals = [w for w, v in vals if abs(v) > 0.1 * v0][:20]
            name = ' '.join(vals)
            self.state.topic_names.append(name)

        return self


class GensimTFIDF(ModelBase):
    name = 'gensim_tfidf'
    def __init__(self, gensim_dictionary=None, **kwargs):
        """
        Wraps around gensim's TFIDF model for the sake of standardization

        Parameters
        ----------
        gensim_dictionary
            gensim's native dictionary, thats enough to make a TFIDF model

        kwargs
            any other kwargs to be passed to parent class __init__
        """
        super(GensimTFIDF, self).__init__(**kwargs)
        self.gensim_dictionary = gensim_dictionary
        self.tfidf = g.models.TfidfModel(dictionary=gensim_dictionary)

    def transform(self, record):
        return self.tfidf[record]


class GensimLDA(StatefulModel):
    name = 'gensim_lda'
    def __init__(self, lda_params: dict, id2word: dict = None, **kwargs):
        """
        Wraps around gensim's LDA model

        Parameters
        ----------
        lda_params
            kwargs to pass to gensim's LDA model initialization

        id2word
            id2word to pass to gensim's LDA model initialization, separate from
            lda_params because id2word needs to be obtained by training a dictionary

        kwargs
            any other kwargs to be passed to parent class __init__
        """
        super(GensimLDA, self).__init__(**kwargs)
        self.params = lda_params
        self.id2word = id2word
        self.state.model = None
        self.state.topic_names = []
        self.state.token2topics = {}
        log.info("Init gensim LDA with params:\n{}".format(self.params))

    def transform(self, record):
        return [(int(ix), float(sc)) for ix, sc in self.state.model[record]]

    def train(self, iterable):
        self.state.model = g.models.LdaMulticore(corpus=iterable, id2word=self.id2word, **self.params)

        # need topic names
        self.state.topic_names = []
        for i in range(self.state.model.num_topics):
            vals = self.state.model.show_topic(i, 100)
            v0 = abs(vals[0][1])
            vals = [w for w, v in vals if abs(v) > 0.1 * v0][:20]
            name = ' '.join(vals)
            self.state.topic_names.append(name)

        # need token2topic (str -> str)
        for i in range(self.state.model.num_terms):
            topics = self.state.model.get_term_topics(i)
            if len(topics) > 0:
                self.state.token2topics[self.state.model.id2word[i]] = [self.state.topic_names[t[0]] for t in topics]

        return self


def norm_gensim_vec(g_vec):
    """
    Normalize a gensim-style vector
    """
    norm_sq = sum([t[1] ** 2 for t in g_vec])
    if norm_sq == 0:
        return g_vec
    norm = norm_sq ** 0.5
    return [(id_, val / norm) for id_, val in g_vec]


def get_gensim_vec_to_list_fn(dims=None):
    """
    creates a function to use in map that transforms a gensim-style vector to just a vector:
    [(0, val_0), (1, val_1), ..., (n, val_n)] --> [val_0, val_1, ..., val_n]

    Parameters
    ----------
    dims
        give number of dimensions if gensim vector is sparse, to make sure the resulting vector is dense

    Returns
    -------
        a vector - list of values
    """
    if dims is None:
        def gensim_vec_to_list(g_vec):
            return [val for _, val in g_vec]
    else:
        def gensim_vec_to_list(g_vec):
            vec = [0] * dims
            for i, val in g_vec:
                vec[i] = val
            return vec
    return gensim_vec_to_list
