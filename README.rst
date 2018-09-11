Cranial Modeling
====================

Generic tools
-------------
- Specialized iter-tools where each iterator can be used more than once, but does not load all data into memory.

Machine Learning Tools
----------------------

- Standard model interface
   ...where "model" means any stateless or stateful
   transformation of data. Because of the standard interface each "model"
   can be used by itself or as a step in another "model". This is similar
   to scikit-learn pipelines, but at a lower level and with a use of specialized
   iter-tools instead of in-memory data.
   
- Standard scripts for deploying Models as microservices
   Including training, and using models for batch or online inference.


Models
------
Contains subclasses of ModelBase. These may have dependencies that must
explicitly be included during pip installation.

Installation
============
The basic installation will include any dependencies needed if you are just writing your own models. 

If you want to use soem of the pre-packaged models in cranial.models, you may need extra packages
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

Usage
======
(Work in progress. Tutorials forthcoming?)

#. Create a directory and change to it.

#. Create a file called `model.py` with a class called `Model` that inherits from `cranial.model_base.ModelBase` or `cranial.model_base.StatefulModel`.

#. Create a `config.json` file with at least `{"model_name": "some_unique_name"}`.

#. Run `python3 -m cranial.service_scripts.serving --no-load`. 

About Cranial
======================

Cranial is a Framework and Toolkit for building distributed applications and
microservices in Python, with a particular focus on services delivering
predictions from online learning models.

The machine learning components do not provide algorithms or models like
SciKitLearn or Tensorflow or Spark or H2O, but instead provide wrappers so that
models and pipelines created by these tools can be deployed and combined in
standardized ways.
