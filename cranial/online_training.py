from abc import abstractmethod, ABCMeta
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from cranial.common import logger
from cranial.model_base import ModelBase, StatefulModel

log = logger.get(name='online_learning')


class TrainerBase(metaclass=ABCMeta):
    """
    Object responsible for defining when and how to update a model

        - `is_ready` is a method that will be called every time a transform method of OnlineLearningWrapper is
            called, if it returns True then OnlineLearningWrapper will try to get a training data from its accumulator
            and use it to update a model, or in case of remote updates will try to load a saved state from a connector.

        - `update` is a method that defines how to update: call a model.update with accumulated data, or
            start to load a remotely stored state
    """

    @abstractmethod
    def update(self, model, data):
        """
        Should take model and training data as arguments and return an updated model. The updating logic can be
        anything, it can use the data, or not use the data, or maybe completely re-instantiates a model. It's up to a
        developer and what needs they have.

        This method should also return True/False whether update was completed, this will allow OnlineLearningWrapper
        to call update again even outside of schedule to check again if update was completed

        Parameters
        ----------
        model
            model that needs to be updated

        data
            data to use for updates

        Returns a tuple
        -------
            (updated or original model, True/False if model was updated)
        """
        return model, True


class AccumulatorBase(metaclass=ABCMeta):
    def __init__(self):
        """
        This object is responsible for accumulating incoming data and organizing it into training examples
        """
        self._batch = []

    @abstractmethod
    def add(self, record):
        """
        Implement logic to add record to accumulator
        """
        pass

    @abstractmethod
    def get_batch(self):
        """
        Implement logic to return data for updates, don't forget to reset batch if it is needed.
        """
        return self._batch

    def reset(self):
        """
        removes all stored examples
        """
        self._batch = []


class ScheduleBase(metaclass=ABCMeta):
    @abstractmethod
    def is_ready(self):
        """
        Implement logic defining when to update, can depend on a number of examples or a time passed...
            or checks if download of saved state was complete
            ...
        """
        return True


class OnlineLearningWrapper(ModelBase):
    def __init__(self, model: StatefulModel, trainer: TrainerBase, schedule: ScheduleBase,
                 accumulator: AccumulatorBase = None, **kwargs):
        """
        This object converts a stateful model to a model that can learn from data passed for inference

        To learn almost-online two things are needed
            - Accumulating incoming data into micro-batches.
                This is be done with accumulator object
            - Specifying what kind of updates to make (use data to do updates, load from a remote storage, etc...).
                This is defined in trainer object.
            - schedule - when to do updates. This is also defined in trainer object

        Parameters
        ----------
        model
            a stateful model to convert into an online learning model

        trainer
            trainer is responsible to accumulating and keeping examples for training and specifies how to update

        accumulator
            an object responsible for accumulating incoming data into batches used for updates. This one is optional,
            the default is None, in that case now incoming data will be stored anywhwere and will not be used for
            models updates. This should be used in case of remote updates form a saved files.

        kwargs
        """
        super(OnlineLearningWrapper, self).__init__(**kwargs)
        self.trainer = trainer
        self.model = model
        self.schedule = schedule
        self.accumulator = accumulator
        # self.name = self.model.name

        self._retry = False
        self._last_update_data = []

    def transform(self, record):
        """
        A higher level composed model (where this wrapper model is just a single step in a chain of transformations)
        will always call `transform` method for inference. Since the goal is to learn online, this modified `transform`
        method should contain both actual transform step and updates to the model if needed.

        Three things happen here
            1. add example to accumulator
            2. update model if it's time or need to check on incomplete previous updates
            3. actual transform of an input data

        Parameters
        ----------
        record
            incoming single record of data that needs to be transformed

        Returns
        -------
            transformed data
        """
        # 1. add record to accum
        if self.accumulator is not None:
            self.accumulator.add(record)

        # 2. maybe update

        # first see if need to re-try previous update, this is in case update consisted of just a
        # future and now need to check again if future was done
        if self._retry:
            self.model, success = self.trainer.update(self.model, self._last_update_data)
            self._retry = not success

        # now see if schedule thinks it's time to update
        if self.schedule.is_ready():
            # get data for updates (could be [], but its ok, model's update should be able to handle that)
            self._last_update_data = [] if self.accumulator is None else self.accumulator.get_batch()

            # trainer updates model, could be based on provided data or not at all
            self.model, success = self.trainer.update(self.model, self._last_update_data)
            # set to re-try if update was unsuccessful
            self._retry = not success

        # 3. finally return the transformation (it does not matter that it is after update at all)
        return self.model.transform(record)

    def load(self, *args, **kwargs):
        """
        pass through to the models's load method
        """
        return self.model.load(*args, **kwargs)

    def save(self, *args, **kwargs):
        """
        pass through to the models's save method
        Parameters
        ----------
        fpath
            file path for saving model's state
        """
        return self.model.save(*args, **kwargs)

    @property
    def name(self):
        return self.model.name

    @name.setter
    def name(self, new_name):
        self.model.name = new_name


##### below are specific implementations for accumulators and trainers


class CountSchedule(object):
    def __init__(self, update_freq, start_true=False):
        """
        Schedule that triggers after a specified number of calls

        Parameters
        ----------
        update_freq
            number of calls that needs to pass before triggering

        start_true
            the very first call will return True, useful in combination with remote loading
        """
        self.update_freq = update_freq
        self.start_true = start_true

        self._counter = -1 if start_true else 0

    def is_ready(self):
        """
        This is called every time a OnlineLearningWrapper.transform is called. This method will count the number of
        times it is called and will return True if it counted to `update_freq`, otherwise False

        Returns
        -------
            True/False whether update need to happen
        """
        self._counter += 1
        return self._counter % self.update_freq == 0


class TimeSchedule(object):
    def __init__(self, update_freq, start_true=False):
        """
        Schedule that triggers after a specified period of time passed

        Parameters
        ----------
        update_freq
            time in seconds that needs to pass before triggering

        start_true
            the very first call will return True, useful in combination with remote loading
        """
        self.update_freq = update_freq
        self.start_true = start_true

        self._last_update_time = time.time()
        self._first_time = start_true

    def is_ready(self):
        """
        Start update process when enough time passed
        Returns
        -------
            True/False if passed time is larger than a given value
        """
        if self._first_time:
            self._first_time = False
            return True
        tmp = time.time() - self._last_update_time > self.update_freq
        if tmp:
            self._last_update_time = time.time()
        return tmp


class SimpleAccumulator(AccumulatorBase):
    def __init__(self, max_size=None):
        """
        A trainer that will update a model when minimum number of examples is accumulated.
        Parameters
        ----------
        max_size
            if None, then accumulator does not have a max size and every time data is used the accumulator will
            be emptied. If not None, then when data is used accumulator will not be emptied, instead a new data will
            replace an oldest data if size reached max_size
        """
        super(SimpleAccumulator, self).__init__()
        self._batch = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, record):
        """
        In this simple accumulator examples of data are just added directly

        Parameters
        ----------
        record
            a new single example of incoming data
        """
        self._batch.append(record)

    def get_batch(self):
        """
        Request a micro-batch of data for updates, it could be an empty list if no examples are
        currently in the accumulator

        Returns
        -------
            a list of data records intended for updating a model
        """
        batch = self._batch
        if self.max_size is None:
            self.reset()
        return batch


class LocalTrainer(TrainerBase):
    def __init__(self, connector=None, wait_future=False):
        """
        This trainer will be ready to update a model after every `update_freq` examples of new data (because is_ready
        is called for every example)

        Parameters
        ----------
        update_freq
            number of examples for inference to see for each update (there is no guarantee that it will be equal
            to a number of examples to be used for an actual update, because a matching accumulator might not have
            any matched examples yet.)

        connector
            every time model is updated, it will be also saved to a location specified by a
            connector (connector.put will be called). Connector should already contain a final destination.
        """
        self.connector = connector
        self.wait_future = wait_future

        # need its own thread pool so that reading connector's buffer and unpickling can happen in a separate thread
        self._pool = ThreadPoolExecutor(1)
        self._future = None

    def update(self, model, data):
        """
        This trainer will use passed data for calling model's update method if data is not an empty list.

        Also if connector is given, it will use it to save state remotely (put to a connector)

        Parameters
        ----------
        model
            a model to update

        data
            data to use for updating the model

        Returns
        -------
            (updated model, True) this kind of trainer always returns success
        """
        if len(data) > 0:
            model = model.update(data)

        # save to a specified location if provided
        if self.connector is not None:
            if self.wait_future:
                model.save(connector=self.connector)
            else:
                self._pool.submit(lambda: model.save(connector=self.connector))

        return model, True


class RemoteLoadTrainer(TrainerBase):
    def __init__(self, connector, wait_future=False):
        """
        Loads a state from a remote location

        Parameters
        ----------
        update_freq
            how often try to load in seconds

        connector
            connector should already have a final remote location where to get saved state

        """
        self.connector = connector
        self.connector.do_read = False
        self.wait_future = wait_future

        # need its own thread pool so that reading connector's buffer and unpickling can happen in a separate thread
        self._pool = ThreadPoolExecutor(1)
        self._future = None

    def update(self, model, _):
        """
        This trainer will start a new thread where loading of new state is done and return a future of its result, and
        if that future already exists then will check if loading was complete, then will swap current model's state
        with new loaded one

        Because this trainer only loads remotely stored state, it does not need any accumulated data.

        Parameters
        ----------
        model
            a model to update

        _
            second argument (passed in data) is not used for remote loading

        Returns
        -------
            (updated model, success indicator)
        """
        success = False
        if self._future is None:
            # this will do everything in the background and will return a state object already loaded into memory
            # self._future = self._pool.submit(lambda: pickle.loads(self.connector.get().read()))
            self._future = self._pool.submit(lambda: model.load(connector=self.connector))

        if self._future.done() or self.wait_future:

            try:
                model.state = self._future.result().state
            except Exception as e:
                log.error("NO UPDATE:\t{}".format(e))

            # reset future only after it is done
            self._future = None
            success = True  # set True even if there was an exception in future because the process was complete

        return model, success
