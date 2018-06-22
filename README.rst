Cranial Modeling
====================

Generic tools
-------------
#. Specialized iter-tools where each iterator can be used more than once, but does not load all data into memory.

Machine Learning Tools
----------------------
#. Standard model interface, where "model" means any stateless or stateful
   transformation of data. Because of the standard interface each "model"
   can be used by itself or as a step in another "model". This is similar
   to scikit-learn pipelines, but at a lower level and with a use of specialized
   iter-tools instead of in-memory data. This is also similar to spark or dask,
   but in pure python and without sofisticated parallelization across machines.
#. Standard scripts for training models and using models for batch or online
   inference.

Models
------
Contains subclasses of ModelBase. These may have dependencies that must
explicitly be included during pip installation.

Installation
============
Please note that pytorch cannot be installed through the normal pip installation and
the developer is responsible for installing it, if desired, on their own.

The basic installation will include any dependencies needed to run code from the
cranial and cranial.fetchers modules. Code from cranial.models may need extra packages
installed. Available extras are:

bpe-nlp                   Installs numpy in support of bpe and nlp.
gensim                    Installs the gensim package to support gensim_models.
p2p_utils                 Installs bs4 in support of p2p_utils.
spacy                     Installs spacy in support or the spacy_tokenizer.


Example of installing spacy and bpe-nlp support
.. code-block:: bash

   pip install "cranial[spacy, bpe-nlp]"


After spacy installation, the additional following step is required:
.. code-block:: bash

   python -m spacy.en.download parser


About Cranial
======================

Cranial is a Framework and Toolkit for building distributed applications and
microservices in Python, with a particular focus on services delivering
predictions from online learning models.

The machine learning components do not provide algorithms or models like
SciKitLearn or Tensorflow or Spark or H2O, but instead provide wrappers so that
models and pipelines created by these tools can be deployed and combined in
standardized ways.
