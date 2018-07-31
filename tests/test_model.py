"""Tests for the models.model module"""
from __future__ import absolute_import, print_function

import numpy as np
from numpy.testing import assert_raises
import six

#import clmm
from clmm import Model, Parameter
#from clmm import models
#from clmm.models import model
#from clmm.models.model import Model
#from clmm.models import parameter
#from clmm.models.parameter import Parameter


def assert_block(test_model):
    """Block of asserts for type checks in models.model

    Parameters
    ----------
    test_model : Model instance
        Instance of the Model class to run asserts on
    """

    assert callable(test_model.func)

    assert (np.iterable(test_model.independent_vars) \
            and not isinstance(test_model.independent_vars, dict)) \
            or test_model.independent_vars is None
    if test_model.independent_vars is not None:
        for element in test_model.independent_vars:
            assert isinstance(element, six.string_types)

    assert (np.iterable(test_model.params) \
            and not isinstance(test_model.params, dict)) \
            or (test_model.params is None)
    if test_model.params is not None:
        for element in test_model.params:
            assert isinstance(element, Parameter)


def test_model_superclass():
    """Test the Model superclass. """

    assert_raises(TypeError, Model, lambda x: x, [1])
    assert_raises(TypeError, Model, lambda x: x, 'r')
    assert_raises(TypeError, Model, 'x*x', ['r'])
    assert_raises(TypeError, Model, lambda x: x, ['r'], ['param1'])

    test_model = Model(lambda x: x)
    assert_block(test_model)

    test_model = Model(lambda x: x, ['r'])
    assert_block(test_model)

    test_model = Model(lambda x: x*x, ['r'], [Parameter()])
    assert_block(test_model)
