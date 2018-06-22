from cranial.messaging import Async_WrapperPool
from cranial.model_base import ModelBase
from abc import ABCMeta, abstractmethod


class ReporterBase(ModelBase, metaclass=ABCMeta):
    name = "metric reporter"

    def __init__(self, apply_funcs, **kwargs):
        """
        This is just a step to be used in a model (or dataset or consumer) transformations chain

        because calculations of metrics will happen in global scope, make metrics as simple as
        possible to avoid slowing down the whole model

        Parameters
        ----------
        apply_funcs
            single or a list of functions that will be calculated for every data record
            usually this should be at least a function that serializes data because only str/bytes can be sent
        kwargs
            passed to a parent class
        """
        super(ReporterBase, self).__init__(**kwargs)
        self.apply_funcs = apply_funcs if isinstance(apply_funcs, list) else [apply_funcs]

    @abstractmethod
    def report(self, record):
        """
        calculate metric and define how to send, messenger.notify or notifier.send, etc...

        Parameters
        ----------
        record
            a current data record
        """
        values = [fn(record) for fn in self.apply_funcs]
        pass

    def transform(self, record):
        """
        This is the tranformation for this step, all it does is report the current
        data record and then returns back what it got

        Parameters
        ----------
        record
            incoming data
        Returns
        -------
            same record
        """
        self.report(record)
        return record


class MessengerReporter(ReporterBase):
    name = "messenger metric reporter"

    def __init__(self, apply_funcs, messenger, **kwargs):
        """
        reporter that uses messenger
        Parameters
        ----------
        apply_funcs
            optional single or a list of functions that will be calculated for every data record
            usually this should be at least a funciton that serializes data because only str/bytes can be sent
        messenger
            messenger that will send messages through all of its notifiers for every incoming data
        kwargs
            passed to a parent class
        """
        super(MessengerReporter, self).__init__(apply_funcs, **kwargs)
        self.messenger = messenger

    def report(self, record):
        [self.messenger.notify(fn(record)) for fn in self.apply_funcs]


class NotifierReporter(ReporterBase):
    name = "notifier metric reporter"

    def __init__(self, apply_funcs, notifier, address, endpoint, n_threads=None, **kwargs):
        """
        reporter that uses a single notifier

        Parameters
        ----------
        apply_funcs
            optional single or a list of functions that will be calculated for every data record
            usually this should be at least a funciton that serializes data because only str/bytes can be sent
        notifier
            notifier that will send messages
        address
            address to use in notifier.send()
        endpoint
            endpoint to use in notifier.send()
        n_threads
            if not none, make sending async with specified number of threads
        kwargs
            passed to a parent class
        """
        super(NotifierReporter, self).__init__(apply_funcs, **kwargs)
        self.notifier = notifier
        self.n_threads = n_threads
        self.address = address
        self.endpoint = endpoint

        # choose which reporting method to use
        if n_threads is not None and (not isinstance(self.notifier, Async_WrapperPool)):
            self.notifier = Async_WrapperPool(notifier, n_threads=self.n_threads)

    def report(self, record):
        [self.notifier.send(message=fn(record), address=self.address, endpoint=self.endpoint)
         for fn in self.apply_funcs]
