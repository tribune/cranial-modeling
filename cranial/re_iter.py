"""
Specialized iter-tools, where each object acts as a generator by yielding one data
element at a time and not loading all avalable data into memory, but iteration can be done more than once.
"""

import time
import uuid
import os
import json
import random

# changing to this one because it uses dill instead of pickle, so can use lambdas closures and functools
from pathos.pools import ProcessPool
from multiprocessing.pool import ThreadPool

from cranial.common import logger

log = logger.get(name='re_iter', var='REITER_LOGLEVEL')


class ReIterBase(object):
    """
    Base class, do not use by itself
    Defines convenience methods
    """

    def __init__(self, iterable_input=None, name='', verbose=True):
        self.iterable_input = iterable_input
        self.name = name
        self.verbose = verbose
        self.iter_counter = 0
        self.item_counter = 0
        self.time_start = None
        self._curr_generator = None

    def _on_iter_start(self):
        self.iter_counter += 1
        if self.verbose:
            log.info("{}:\tStart iter number {}".format(self.name, self.iter_counter))
        self.time_start = time.time()
        self.item_counter = 0

    def _on_iter_end(self):
        if self.verbose:
            log.info("{}:\tFinished iter number {}\ttotal items: {}\ttotal time: {:.1f} sec".format(
                self.name, self.iter_counter, self.item_counter, time.time() - self.time_start))

    def _iter(self):
        raise NotImplementedError

    def __iter__(self):
        self._on_iter_start()
        self._curr_generator = self._iter()
        for itm in self._curr_generator:
            yield itm
            self.item_counter += 1
        self._on_iter_end()

    def __len__(self):
        """
        It is better not to call length at the end of a chain, because it will
        cause to go through all data and perform all transformation steps

        Alternatively this can be modified to
        ```
        try:
            return len(self.input_iterator)
        except:
            return [1 for _ in self.input_iterator]
        ```
        but that will return potentially wrong number since iterator can modify number of items
        """
        return sum([1 for _ in self])


class ReGenerator(ReIterBase):
    def __init__(self, one_time_generator_fn, name='reGenerate', verbose=True):
        """
        Given a function that returns a one-time generator, converts it into a many-times iterator

        Parameters
        ----------
        one_time_generator_fn
            a function that returns a generator
        name
            name to use for logging messages,
        """
        super().__init__(name=name, verbose=verbose)
        self.one_time_generator_fn = one_time_generator_fn

    def _iter(self):
        return self.one_time_generator_fn()


class ReMap(ReIterBase):
    def __init__(self, fn, iterable_input, proc_type=None, n_proc=1, per_proc_buffer=1, ordered=True, name='reMap', verbose=True):
        """
        This is a map function that can be iterated over more than once. Returns an iterator.

        Parameters
        ----------
        fn
        iterable_input
            iterable input

        proc_type
            if 'sub' then uses a pathos ProcessPool to map function
            if 'thread' then uses standard multiprocessing ThreadPool
            else uses regular map

        n_proc
            number of workers in a pool (ignored if no pool)

        per_proc_buffer
            since pool's map function does not know limits, there is a forced stop-and-yield-all after
            this many processed tasks per process/thread

        ordered
            use ordered map by default, uses `imap_unordered` otherwise

        name
            name to use for logging messages

        verbose
        """
        name += '' if proc_type not in ('sub', 'proc', 'subprocess', 'th', 'thread') else ' ' + proc_type
        super().__init__(iterable_input=iterable_input, name=name, verbose=verbose)
        self.fn = fn
        self.proc_type = proc_type
        self.per_proc_buffer = per_proc_buffer
        self.n_proc = n_proc
        self.ordered = ordered

    def _iter(self):
        if self.proc_type in ('thread', 'th') and self.n_proc > 0:
            with ThreadPool(self.n_proc) as p:
                # this is a workaround for limiting input iterator consumption, got it from SO
                buff = []
                for itm in self.iterable_input:
                    buff.append(itm)
                    if len(buff) >= self.per_proc_buffer * self.n_proc:

                        if self.ordered:
                            for itm in p.imap(self.fn, buff):
                                yield itm
                        else:
                            for itm in p.imap_unordered(self.fn, buff):
                                yield itm
                        buff = []

                # feed the remaining buffer after input is exhausted
                if self.ordered:
                    for itm in p.imap(self.fn, buff):
                        yield itm
                else:
                    for itm in p.imap_unordered(self.fn, buff):
                        yield itm

        elif self.proc_type in ('sub', 'proc', 'subprocess') and self.n_proc > 0:
            try:
                log.info("Trying to terminate previous pool")
                # this is stupid, but that's how pathos is built
                self.pool.terminate()
                self.pool.clear()
                log.info("Yay! Cleared previous process pool")
            except AttributeError:
                log.warning("Is this the first time creating a pool...")

            self.pool = ProcessPool(nodes=self.n_proc)

            # this is a workaround for limiting input iterator consumption, got it from SO
            buff = []
            for itm in self.iterable_input:
                buff.append(itm)
                if len(buff) >= self.per_proc_buffer * self.n_proc:
                    if self.ordered:
                        for itm in self.pool.imap(self.fn, buff):
                            yield itm
                    else:
                        for itm in self.pool.uimap(self.fn, buff):
                            yield itm
                    buff = []

            # feed the remaining buffer after input is exhausted
            if self.ordered:
                for itm in self.pool.imap(self.fn, buff):
                    yield itm
            else:
                for itm in self.pool.uimap(self.fn, buff):
                    yield itm

        else:
            for itm in map(self.fn, self.iterable_input):
                yield itm


class ReFilter(ReIterBase):
    def __init__(self, fn, iterable_input, name='reFilter', verbose=True):
        """
        Filter function that can be used more than once, returns an iterator

        Parameters
        ----------
        iterable_input
            iterable input

        fn
            filter function

        name
            name to use for logging messages
        """
        super().__init__(iterable_input=iterable_input, name=name, verbose=verbose)
        self.fn = fn

    def _iter(self):
        return filter(self.fn, self.iterable_input)


class ReDedup(ReIterBase):
    def __init__(self, iterable_input, dedup_key, name='reFilter', verbose=True):
        """
        This iterator keeps track of seen values of a certain field in items (of type dict)
        and yield only yet unseen ones

        The reason this object is here instead of in models is because
            1. it does not save the state anywhere
            2. it does not return an output for every input

        Parameters
        ----------
        iterable_input
            iterable input

        dedup_key
            a key of dict for which to track values

        name
            name to use for logging messages
        """
        super().__init__(iterable_input=iterable_input, name=name, verbose=verbose)
        self.dedup_key = dedup_key
        self._known_values = set()

    def _iter(self):
        self._known_values = set()
        for itm in self.iterable_input:
            val = itm.get(self.dedup_key)
            if val in self._known_values:
                continue
            else:
                self._known_values.add(val)
                yield itm


class ReChain(ReIterBase):
    """
    Analog of itertools.chain that can be iterated multiple times

    Given an iterator where each item is iterable itself, returns a single iterator with all sub-items
    """

    def _iter(self):
        for itm in self.iterable_input:
            for sub_itm in itm:
                yield sub_itm


class ReRepeat(ReIterBase):
    def __init__(self, iterable_input, n=1, name='re-repeat', verbose=True):
        """
        NOT an analog of itertools.repeat
        Returns an multi-use iterator where each item of an input iterator is repeated n times.
        Example:
            given input = [1, 2, 3] and n = 2 -> [1, 1, 2, 2, 3, 3]

        Parameters
        ----------
        iterable_input
            input iterable

        n
            how many times to repeat each item

        name
            name to use for logging messages
        """
        super().__init__(iterable_input=iterable_input, name=name, verbose=verbose)
        self.n = n

    def _iter(self):
        for itm in self.iterable_input:
            for _ in range(self.n):
                yield itm


class ReCycle(ReIterBase):
    def __init__(self, iterable_input, n=0, name='re-cycle', verbose=True):
        """
        Analog of itertools.cycle, but can be iterated over multiple times
        Returns an iterator that repeats input sequence n times, or infinite number of times if n = 0

        Parameters
        ----------
        iterable_input
            iterable input

        n
            number of time to repeat input

        name
            name to use for logging messages
        """
        super().__init__(iterable_input=iterable_input, name=name, verbose=verbose)
        self.n = n

    def _iter(self):
        n_ = 0
        while True:
            for itm in self.iterable_input:
                yield itm
            n_ += 1
            if n_ == self.n:
                break


class ReZip(ReIterBase):
    def __init__(self, *iterable_inputs, name='re-zip', verbose=True):
        """
        Analog of zip, but can be iterated more than once

        Parameters
        ----------
        iterable_inputs
            iterable input (each item is a tuple to zip)

        name
            name to use for logging messages
        """
        super().__init__(iterable_input=iterable_inputs, name=name, verbose=verbose)

    def _iter(self):
        return zip(*self.iterable_input)


class DiskCache(ReIterBase):
    def __init__(self, iterable_input, name='Disk Cache', tmp_file_path=None, tmp_file_dir=None,
                 serializer='json', delete_when_done=True, verbose=True):
        """
        On the first pass the results are stored in a temp file, on all subsequent passes input_iterator is
        not used, instead results are read from temp file.

        Parameters
        ----------
        iterable_input
            input objects

        name
            for logging messages

        tmp_file_path
            file to save iterable

        serializer
            how to serialise objects to file, default 'json'

        delete_when_done
            if True, deletes file when object is deleted
        """
        super().__init__(iterable_input=iterable_input, name=name, verbose=verbose)
        self.delete_when_done = delete_when_done

        self.serializer = serializer
        assert self.serializer in ['json'], "Unsupported serializer"

        self.tmp_filename = str(uuid.uuid4()) if tmp_file_path is None else tmp_file_path
        if tmp_file_dir is not None:
            self.tmp_filename = os.path.join(tmp_file_dir, self.tmp_filename)
        dir_path = os.path.split(self.tmp_filename)[0]
        if len(dir_path) > 0:
            os.makedirs(dir_path, exist_ok=True)
        self.cleanup()  # in case file already exists

        self.cached = False
        self.file_size = 0

    def _yield_write_json(self):
        log.info("{}:\tSaving iterable to {}".format(self.name, self.tmp_filename))
        with open(self.tmp_filename, 'w') as f:
            for res in self.iterable_input:
                try:
                    self.file_size += f.write(json.dumps(res) + '\n')
                except:
                    log.info(res)
                    raise
                yield res
        self.cached = True
        log.info("{}:\tSaved iterable to {}, size {:,}".format(self.name, self.tmp_filename, self.file_size))

    def _yield_read_json(self):
        log.info("{}:\tReading saved iterable from {}".format(self.name, self.tmp_filename))
        with open(self.tmp_filename, 'r') as f:
            for line in f:
                yield json.loads(line)

    def _iter(self):
        if not self.cached:
            if not os.path.isfile(self.tmp_filename):
                return self._yield_write_json()
            else:
                return self.iterable_input
        else:
            return self._yield_read_json()

    def cleanup(self):
        """
        remove temp file
        """
        if os.path.isfile(self.tmp_filename):
            os.unlink(self.tmp_filename)

    def __del__(self):
        if self.delete_when_done:
            self.cleanup()


class ReBatch(ReIterBase):
    def __init__(self, iterable_input, batch_size, only_full=False, shuffle=False, buffer_size=None,
                 name='reBatch', verbose=True):
        """
        combine items from input iterator into batches

        Parameters
        ----------
        iterable_input
            iterator with individual items

        batch_size
            batch size

        only_full
            if True, skip last batch that has less than batch_size items

        shuffle
        buffer_size

        name
            name to use for logging
        """
        super().__init__(iterable_input=iterable_input, name=name, verbose=verbose)
        self.batch_size = batch_size
        self.only_full = only_full
        self.shuffle = shuffle
        self.buffer_size = self.batch_size if buffer_size is None else buffer_size

    def _iter(self):
        buffer = []
        i = 0
        for itm in self.iterable_input:
            buffer.append(itm)
            i += 1
            if i == self.buffer_size:
                if self.shuffle:
                    random.shuffle(buffer)

                # release one or more full batches
                n_batches = self.buffer_size // self.batch_size
                if n_batches > 1:
                    # yield only half of available batches
                    # intended for use with buffer_size >> batch_size and shuffle=True
                    n_batches = n_batches // 2
                for i in range(n_batches):
                    batch = buffer[i * self.batch_size: (i + 1) * self.batch_size]
                    yield batch
                buffer = buffer[self.batch_size * n_batches:]
                i = len(buffer)

        # yielding remaining
        if self.shuffle:
            random.shuffle(buffer)

        # number of full batches
        n_batches = len(buffer) // self.batch_size
        for i in range(n_batches):
            batch = buffer[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch

        if not self.only_full and len(buffer) > n_batches * self.batch_size:
            batch = buffer[n_batches * self.batch_size:]
            yield batch


class BucketBatch(ReIterBase):
    def __init__(self, iterable_input, batch_size, buckets, pad_index, only_full=False, field=None,
                 shuffle=False, buffer_size=None, name='Bucket Batch', verbose=True):
        """
        combine items from input iterator into batches

        Parameters
        ----------
        iterable_input
            iterator with individual items

        batch_size
            batch size

        buckets
            list of integers - quantized lengths of sequences,
            for example [8, 16, 32, 64] means no sequence below length 4, or above 64

        pad_index
            index to pad with

        only_full
            if True, skip last batch that has less than batch_size items

        name
            name to use for logging
        """
        super().__init__(iterable_input=iterable_input, name=name, verbose=verbose)
        self.batch_size = batch_size
        self.buckets = buckets
        self.max_length = buckets[-1]
        self.pad_index = pad_index
        self.only_full = only_full
        self.field = field
        self.shuffle = shuffle
        self.buffer_size = self.batch_size if buffer_size is None else buffer_size

    def _iter(self):
        bucket_batches = {b: [] for b in self.buckets}
        bucket_sizes = {b: 0 for b in self.buckets}

        for itm in self.iterable_input:
            if self.field is not None:
                if isinstance(itm, tuple):
                    itm = list(itm)
                real_itm = itm
                itm = itm[self.field]
            length = len(itm)
            if length == 0:
                continue
            if length > self.max_length:
                itm = itm[:self.max_length]
                length = self.max_length
            for lower_b, upper_b in zip(self.buckets, self.buckets[1:]):
                if lower_b < length <= upper_b:
                    num_pad = upper_b - length
                    itm += [self.pad_index] * num_pad
                    if self.field is not None:
                        real_itm[self.field] = itm
                        itm = real_itm
                    bucket_batches[lower_b].append(itm)
                    bucket_sizes[lower_b] += 1
                    ##########################
                    # yielding
                    if bucket_sizes[lower_b] == self.buffer_size:
                        buffer = bucket_batches[lower_b]
                        if self.shuffle:
                            random.shuffle(buffer)

                        # release one or more full batches
                        n_batches = self.buffer_size // self.batch_size
                        if n_batches > 1:
                            n_batches = n_batches // 2

                        for i in range(n_batches):
                            batch = buffer[i * self.batch_size: (i + 1) * self.batch_size]
                            yield batch
                        bucket_batches[lower_b] = buffer[self.batch_size * n_batches:]
                        bucket_sizes[lower_b] = len(bucket_batches[lower_b])
                    ##########################

                    # since this item was placed, stop the loop
                    break

        ##########################
        # yielding remaining
        for lower_b in self.buckets[:-1]:
            buffer = bucket_batches[lower_b]
            if self.shuffle:
                random.shuffle(buffer)

            # number of full batches
            n_batches = len(buffer) // self.batch_size
            for i in range(n_batches):
                batch = buffer[i * self.batch_size: (i + 1) * self.batch_size]
                yield batch

            # check if there is last batch and it needs to be yielded
            if not self.only_full and len(buffer) > n_batches * self.batch_size:
                batch = buffer[n_batches * self.batch_size:]
                yield batch


class Progress(ReIterBase):
    def __init__(self, iterable_input, max_period=2000, name='progress', verbose=True):
        """Utility for logging number of processed items

        Example:
        # >>> a = range(20)
        # >>> b = Progress(a, max_period=7)
        # >>> _ = [_ for _ in b]
        # 2018-02-09T14:46:05PST - re_iter.py - INFO - progress:  Start iter number 5
        # 2018-02-09T14:46:05PST - re_iter.py - INFO - progress yielded 1 items
        # 2018-02-09T14:46:05PST - re_iter.py - INFO - progress yielded 2 items
        # 2018-02-09T14:46:05PST - re_iter.py - INFO - progress yielded 5 items
        # 2018-02-09T14:46:05PST - re_iter.py - INFO - progress yielded 7 items
        # 2018-02-09T14:46:05PST - re_iter.py - INFO - progress yielded 14 items
        # 2018-02-09T14:46:05PST - re_iter.py - INFO - progress:  Finished iter number 5  total items: 20 total time: 0.0 sec
        # Out[10]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]


        """
        super().__init__(iterable_input=iterable_input, name=name, verbose=verbose)
        self.max_period = max(max_period, 5)
        self.checkpoints = []
        k = 1
        while k < max_period:
            self.checkpoints.extend([k, 2 * k, 5 * k])
            k *= 10
        self.checkpoints = [ch for ch in self.checkpoints if ch < max_period]

    def _iter(self):
        t_start = time.time()
        i_start = 0
        ema_speed = 0
        for i, itm in enumerate(self.iterable_input):
            if (i + 1) in self.checkpoints:
                log.info("{} yielded {} items".format(self.name, i + 1))
            elif (i + 1) % self.max_period == 0:
                t_now = time.time()
                speed_now = (i - i_start) / (t_now - t_start)
                ema_speed = speed_now if ema_speed == 0 else (0.9 * ema_speed + 0.1 * speed_now)
                t_start = t_now
                i_start = i
                log.info("{} yielded {} items.\tspeed now {:.2f}\tEMA speed {:.2f}".format(
                    self.name, i + 1, speed_now, ema_speed))
            yield itm
