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
The basic installation will include any dependencies needed if you are just
writing your own models.

If you want to use some of the pre-packaged models in cranial.models, you may
need additional packages installed, such as `gensim` or `spacy`.


Example of installing everything:

.. code-block:: bash

   pip install "cranial-modeling spacy gensim"


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

A slide deck with detailed diagrams of Cranial architecture can be found here:
https://docs.google.com/presentation/d/131RK79w-Ls7uKuQocDcyEBXWDWABv6fXpaK_1THBG2Y/edit?usp=sharing

Contributing
============
Questions, Suggestions, Support requests, troubel reports, and of course, 
Pull Requests, are all welcome in the Github issue queue.
