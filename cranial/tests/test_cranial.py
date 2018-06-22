"""Provides doctests for unittest discovery."""

import doctest
from cranial import model_base

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(model_base))
    return tests
