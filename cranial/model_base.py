"""
base classes for models
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import dill as pickle
import os
from collections import OrderedDict
from cranial.common import logger
from cranial.re_iter import ReMap

log = logger.get(name='model_base', var='MODELS_LOGLEVEL')  # streaming log


class State(metaclass=ABCMeta):
    """ An object with save() & load() methods.

    Notice that `Foo` below does NOT inherit from State.
    >>> class Foo():
    ...     def save(self):
    ...             return
    ...     def load(self):
    ...             return
    ...
    >>> isinstance(Foo(), State)
    True
    >>> isinstance(tuple(), State)
    False
    """

    def __str__(self):
        ss = []
        for attr in dir(self):
            if (not attr.startswith("_")) and (not hasattr(getattr(self, attr), '__call__')):
                attr_obj = getattr(self, attr)
                if isinstance(attr_obj, np.ndarray):
                    attr_obj = "arr {} {}".format(attr_obj.shape, attr_obj)
                elif hasattr(attr_obj, "__len__"):
                    attr_obj = "len {} {}".format(len(attr_obj), attr_obj)
                ss.append("{} = {}".format(attr, attr_obj)[:200])
        return '\n'.join(ss)

    def save(self, fpath:str=None, connector=None):
        """
        For now this just pickles the state into a file or puts into a connector stream

        Parameters
        ----------
        fpath
            direct path to a pickled file, or a `name` argument to pass to connector. In the latter case
                this name=fpath string will be appended after '/' separator to the base_address of the connector. For
                example, if connector.base_address = 'some/path' then the final destination will be 'some/path/fpath'.
                Note, fpath can be None only if connector is given and its base_address is a full file path

        connector
            An optional connector object, if None, then state will be saved to a file
        """
        assert (fpath is not None) or (connector is not None), "either fpath or connector should be given"
        if connector is not None:
            log.info("Trying to save using {} to fpath={}".format(connector, fpath))
            connector.put(source=pickle.dumps(self), name=fpath)
            log.info("Saved using {} to fpath={}".format(connector, fpath))
        else:
            log.info("Trying to save {}".format(fpath))
            with open(fpath, 'wb') as f:
                pickle.dump(self, f)
            log.info("Saved {}".format(fpath))

    @classmethod
    def load(cls, fpath:str=None, connector=None):
        """
        Loads pickled state from file or a stream and returns it

        Parameters
        ----------
        fpath
            direct path to a pickled file, or a `name` argument to pass to connector. In the latter case
                this name=fpath string will be appended after '/' separator to the base_address of the connector. For
                example, if connector.base_address = 'some/path' then the final destination will be 'some/path/fpath'.
                Note, fpath can be None only if connector is given and its base_address is a full file path

        connector
            An optional connector object, if None, then state will be loaded from a file

        Returns
        -------
            loaded state object
        """
        assert (fpath is not None) or (connector is not None), "either fpath or connector should be given"
        if connector is not None:
            assert not connector.do_read, "connector should not be in read mode (should return only a readable buffer)"
            assert connector.binary, "connector should be in binary mode"

            log.info("Trying to load using {} from fpath={}".format(connector, fpath))
            obj = pickle.loads(connector.get(name=fpath).read())
            log.info("Loaded using {} from fpath={}".format(connector, fpath))
        else:
            log.info("Trying to load {}".format(fpath))
            with open(fpath, 'rb') as f:
                obj = pickle.load(f)
            log.info("Loaded {}".format(fpath))
        return obj

    @classmethod
    def __subclasshook__(cls, ClassObject):
        """This hook cases `isinstance(x, State)` to be True for any x which is
        an object or class having both save() and load() methods."""
        if cls is State:
            if any("save" in B.__dict__ and "load" in B.__dict__
                   for B in ClassObject.__mro__):
                return True
        return NotImplemented


class ModelBase(metaclass=ABCMeta):
    """A model that does not have a state, just implements a data transformation
    method.

    It still can have some class attributes that are temporary and do not need
    to be stored or modified

    >>> class Foo():
    ...     def transform(self, record):
    ...             return
    ...     def itransform(self, **params):
    ...             return
    ...
    >>> isinstance(Foo(), ModelBase)
    True
    >>> isinstance(tuple(), ModelBase)
    False
    """
    name = 'ModelBase'
    def __init__(self, **kwargs):
        self.proc_type = kwargs.pop('proc_type', None)
        self.n_proc = kwargs.pop('n_proc', 1)
        self.per_proc_buffer = kwargs.pop('per_proc_buffer', 1)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def transform(self, record):
        """
        A method to transform input data, one data-point/row/features/example at
        a time.

        Parameters
        ----------
        record
            a single example of the data
        Returns
        -------
            transformed data
        """
        return record

    def itransform(self, iterable, iter_name=None):
        iter_name = self.__class__.__name__ if iter_name is None else iter_name
        return ReMap(iterable_input=iterable, fn=self.transform, name=iter_name,
                     proc_type=self.proc_type, n_proc=self.n_proc, per_proc_buffer=self.per_proc_buffer)

    @classmethod
    def __subclasshook__(cls, ClassObject):
        if cls is ModelBase:
            if any("transform" in B.__dict__ and "itransform" in B.__dict__
                   for B in ClassObject.__mro__):
                return True
        return NotImplemented


class StatefulModel(ModelBase, metaclass=ABCMeta):
    """
    A BaseModel that needs a state, which needs to be trained.
    Trained state also needs to be saved/loaded into/from a file

    >>> class Foo(ModelBase):
    ...     state = 1
    ...     def transform(self, record):
    ...         return
    ...     def train(self, iter):
    ...         return
    ...
    >>> isinstance(Foo(), StatefulModel)
    True
    >>> isinstance(tuple(), StatefulModel)
    False
    """
    name = 'StatefulModel'
    def __init__(self, **kwargs):
        """
        When defining a specific model - set its default name to something appropriate for that model
        """
        super(StatefulModel, self).__init__(**kwargs)
        self.state = State()

    @abstractmethod
    def train(self, iterable):
        """
        method that modifies the state and returns self
        Parameters
        ----------
        iterable
            data to use for training

        Returns
        -------
            self
        """
        return self

    def save(self, fpath:str=None, connector=None):
        """
        A method to save the state

        Parameters
        ----------
        fpath
            file path, or stream address, to save model's state into, if None, then self.name will be used as file path

        connector
            An optional connector object, if not None the pickled state will be put into that connector and
            fpath (or the self.name) will be added to the connector's base_address. For example, if
            connector.base_path = 'some/path', then final destination will be 'some/path/fpath'
        """
        fpath = self.name if fpath is None else fpath
        self.state.save(fpath=fpath, connector=connector)

    def load(self, fpath:str=None, connector=None):
        """
        a method to load model state from a file or a connector

        Parameters
        ----------
        fpath
            file path, or stream address, to load model's state from, if None, then self.name will be used as file path
            appending '/'+self.name, if a direct file path desired, set append_name=False

        connector
            An optional connector object, if not None the pickled state will be read from that connector and
            fpath (or the self.name) will be added to the connector's base_address. For example, if
            connector.base_path = 'some/path', then final destination will be 'some/path/fpath'

        Returns
        -------
            self
        """
        fpath = self.name if fpath is None else fpath
        self.state = self.state.load(fpath=fpath, connector=connector)
        return self

    @classmethod
    def __subclasshook__(cls, ClassObject):
        if cls is StatefulModel:
            if any("train" in B.__dict__ and
                           issubclass(B, ModelBase) and
                           hasattr(B, 'state')
                   for B in ClassObject.__mro__):
                return True
        return NotImplemented


class ComposedModel(ModelBase):
    """
    WORK IN PROGRESS
    a model that is composed of other models

    I would like to move towards a direction where there is a list (OrderedDict) of
    steps defined and then
        - make_transform will be standard and will just take those steps in order and compose
        - save/load will also go through steps and call save/load if it exsists

    """
    steps = OrderedDict()
    step_setups = OrderedDict()
    name = 'ComposedModel'

    def transform(self, record):
        raise Exception("model composition was not complete")

    @abstractmethod
    def _make_transform(self):
        """
        This defines what composed transform should be like.
        Even though most of the time its going to be a list of functions composed together, I am living this undefined
        for additional flexibility for things like
            composed_transform(x) = f1(f2(f1(x)))
        """
        # theoretically this stuff should work,
        # but for now it is just an idea expressed in code, and the method is left abstract
        transform_steps = []
        for name, step in self.steps.items():
            if callable(step):
                fn = step
            elif isinstance(step, ModelBase):
                fn = step.transform
            else:
                raise Exception("step should be either callabale or a subclass of ModelBase")

            setup = self.step_setups.get(name)
            if setup is None:
                pass
            else:
                fn = setup.get('modifier', lambda x, **kwargs: x)(**setup.get('kwargs', {}))

            transform_steps.append(fn)
        pass

    def save(self, fpath=None, connector=None):
        """
        A master save method that saves all stateful step-models

        Parameters
        ----------
        fpath
            a root path for the whole model, a directory with overall model name will be created at this location

        connector
            Optional connector, if given model will be saved to connector, if not, then model will save to disk at
            the specified `fpath` location
        """
        for name, step in self.steps.items():
            # cannot just check for Stateful, because of the onlineLearningWrapper
            if isinstance(step, ModelBase) and hasattr(step, 'save'):
                step_path = os.path.join(fpath, step.name) if connector is None else fpath
                step.save(fpath=step_path, connector=connector)

    def load(self, fpath=None, connector=None):
        """
        A master load method that loads all stateful step-models

        Parameters
        ----------
        fpath
            a root path for the whole model, a directory with overall model name should exist at this location

        connector
            Optional connector, if given model will be loaded form connector, if not, then model will try to load
            from disk at the specified `fpath` location
        """
        for name, step in self.steps.items():
            # cannot just check for Stateful, because of the onlineLearningWrapper
            if isinstance(step, ModelBase) and hasattr(step, 'load'):
                step_path = os.path.join(fpath, step.name) if connector is None else fpath
                self.steps[name] = step.load(fpath=step_path, connector=connector)

        self._make_transform()

    def _prepend_names(self):
        """
        A must run after steps are set up, otherwise save/load will have wrong paths
        """
        for name, step in self.steps.items():
            if isinstance(step, ModelBase) and not isinstance(step, ComposedModel):
                step.name = self.name + '/' + step.name
                self.steps[name] = step
