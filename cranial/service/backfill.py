"""
Universal backfill for cranial models

- Download and load saved model
- Instantiate raw data stream
- Instantiate consumer / results data putter
- run raw_data->model->consumer

Usage:
    backfill.py --start-time=<s> --end-time=<s> [--config=<s>] [--model_suffix=<s>]

Options:
--start-time=<s>    backfill starting date
--end-time=<s>      backfill ending date
--config=<s>        use a different config file.
                    Can also use a different config file stored in s3, use standard format s3://BUCKET/KEY
                    WARNING: if you decide to deploy models that differ from each
                    other by config only, make sure you also set different model names
--model_suffix=<s>  same config but modify model name by adding a suffix
"""
from docopt import docopt
import os
import traceback
import json

from cranial.fetchers import S3InMemoryConnector
from cranial.fetchers.s3 import from_s3
from cranial.common import logger


# these are supposed to exist at the target location
from model import Model, Dataset, Consumer, BUCKET, MODEL_PREFIX

log = logger.get(name='backfill_job', var='BACKFILL_LOGLEVEL')  # streaming log


def backfill(opts, **kwargs):
    """
    processes data by model
    Parameters
    ----------
    opts
        options dictionary, should have
            'input_data_params' - parameters to instatiate a dataset taht creates a stream of raw data to process,
            'model_params' - model parameters needed to instantiate it
            'output_data_params' - parameters of a consumer of processed data, things like params for additional
                                    post-model tranformations, parameters of destination for outputing results, etc...
    """

    # load model, init dataset and consumer
    m = Model(name=opts['model_name'], **opts['model_params'], **kwargs)
    connector = S3InMemoryConnector(bucket=BUCKET, prefix=MODEL_PREFIX)
    m.load(connector=connector)

    records = Dataset(**opts['input_data_params']).get_data()
    c = Consumer(name=opts['model_name'], **opts['output_data_params'])

    # define model output
    # m.proc_type = 'proc'  # cannot do that because of spacy, even dill does not pickle spacy instances :(
    # m.n_proc = 4          # but it would have worked if no cython functions
    outputs = m.itransform(records)

    # consume results
    c.consume(outputs)


if __name__ == "__main__":
    log.info("STARTING BACKFILL")
    opts = docopt(__doc__)

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

    # merge
    CONFIG['input_data_params']['start_time'] = opts['--start-time']
    CONFIG['input_data_params']['end_time'] = opts['--end-time']

    # clean dirs and run main backfill
    try:
        backfill(CONFIG)
    except Exception as e:
        log.error("Something failed...")
        log.error(traceback.format_exc())
