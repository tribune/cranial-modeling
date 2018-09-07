#!/usr/bin/env python
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

gensim = [
    'gensim'
]

spacy = [
    'spacy'
]

all_models = gensim + spacy

setup(name='cranial-modeling',
      version='0.2.0',
      namespace_packages=['cranial'],
      description='Cranial Modeling',
      long_description=long_description,
      author='Tronc Data Team',
      author_email='merekhinsky@tribuneinteractive.com',
      url='https://github.com/tribune/cranial-modeling',
      packages=find_packages(exclude=['tests*']),
      install_requires=['cranial-messaging',
                        'cranial-datastore',
                        'cranial-common',
                        'docopt',
                        'pathos'],
      extras_require={'all': all_models,
                      'gensim': gensim,
                      'spacy': spacy
                      }
      )
