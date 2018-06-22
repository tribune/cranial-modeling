"""
Universal training for cranial models.

There should be a model.py file in which these three objects are defined:
1.  Dataset(keys_list:(str, tuple, list), **other_params)
    with method get_data() which returns an instance of ReIter subclass with all the data that is needed for training.
2.  Model(**kwargs) with methods
        - train(iter_data)
        - save(s3_connector), which saves the trained model to s3

By passing a --update flag and provide dates model can be updated instead of
training from scratch. But model has to support that.

Usage:
    train.py [--update START_TIME END_TIME] [--config=<s>] [--model_suffix=<s>]

Options:
--config=<s>        use a different config file.
                    Can also use a different config file stored in s3, use standard format s3://BUCKET/KEY
                    WARNING: if you decide to deploy models that differ from each
                    other by config only, make sure you also set different model names
--model_suffix=<s>  same config but modify model name by adding a suffix
--update            update instead of train from scratch
# START_TIME        start date for update data
# END_TIME          end date for update data
"""
from docopt import docopt
import arrow
import os
import traceback
import json

from cranial.common import logger
from cranial.fetchers import S3InMemoryConnector
from cranial.fetchers.s3 import from_s3

# these are supposed to exist at the target location
from model import Model, Dataset, BUCKET, MODEL_PREFIX

log = logger.get(name='train', var='TRAIN_LOGLEVEL')  # streaming log


def train(opts, **kwargs):
    """
    Train a model
    Parameters
    ----------
    opts
        a config - nested dictionary with options

    Returns
    -------
    """
    m = Model(name=opts['model_name'], **opts['model_params'], **kwargs)
    connector = S3InMemoryConnector(bucket=BUCKET, prefix=MODEL_PREFIX)

    # instantiate dataset and data iterator
    dataset = Dataset(**opts['input_data_params'])
    records = dataset.get_data()

    # check data
    if check_data(records, getattr(dataset, 'validate_itm', None)):
        log.info("DATA OK")
    else:
        log.info("BAD DATA")
        return

    if opts.get('--update', False):
        m.load(connector=connector)
        m.update(records)
    else:
        m.train(records)

    log.info("DONE")

    # save model
    m.save(connector=connector)


def check_data(iter_data, validation_fn=None):
    # test that there is at least one item and its a string
    try:
        itm = next(iter(iter_data))
        return True if validation_fn is None else validation_fn(itm)
    except StopIteration:
        log.error("got an empty data iterator")
        return False


if __name__ == "__main__":

    opts = docopt(__doc__)
    log.info("train script got this options:\n{}".format(opts))

    # maybe use different config file
    config_filename = opts['--config'] if opts['--config'] else 'config.json'
    if config_filename.startswith('s3://'):
        # this will download file to the current dir using aws CLI
        from_s3(config_filename, '.')
        log.info("downloaded config file from {}".format(config_filename))
        _, config_filename = os.path.split(config_filename)

    log.info("using config file: " + config_filename)
    with open(config_filename) as f:
        CONFIG = json.load(f)

    # maybe add a suffix to the model name
    CONFIG['model_name'] += opts['--model_suffix'] if opts['--model_suffix'] else ''

    CONFIG['--update'] = opts['--update']

    if CONFIG['--update']:
        ed = arrow.utcnow().ceil('day') if opts['END_TIME'] is None else arrow.get(opts['END_TIME'])
        sd = ed.shift(days=-1) if opts['START_TIME'] is None else arrow.get(opts['START_TIME'])
        CONFIG['input_data_params']['start_time'] = str(sd)
        CONFIG['input_data_params']['end_time'] = str(ed)

    try:
        train(CONFIG)
    except Exception as e:
        log.error("Something failed during training...")
        log.error(traceback.format_exc())
    finally:
        pass
