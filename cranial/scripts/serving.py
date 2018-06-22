"""
Standard script for serving prediction that use cranial.

Usage:
    serving.py [--demo] [--port=PORT | --topic=TOPIC] [--no-load] [--config=<s>] [--model_suffix=<s>]

Options:
--demo              run demo mode
--port=PORT         Port for ZMQ listener. [default: 5678]
--topic=TOPIC       Topic for Kafka listener.
--no-load           if set, model will not load from previously saved state
--config=<s>        use a different config file.
                    Can also use a different config file stored in s3, use standard format s3://BUCKET/KEY
                    WARNING: if you decide to deploy models that differ from each
                    other by config only, make sure you also set different model names
                    UNLESS different configs are for differentiating between updater and
                    predictor nodes, then model name should be the same
--model_suffix=<s>  same config but modify model name by adding a suffix
"""
# Core modules.
import json
import time
import os

# 3rd-party modules.
from docopt import docopt

# 1st-party modules.
from cranial.common import logger
from cranial.fetchers import S3InMemoryConnector
from cranial.fetchers.s3 import from_s3
from cranial.listeners.zmq import Zmq
from cranial.listeners import Demo

# these are supposed to exist at the target location
from model import Model, BUCKET, MODEL_PREFIX

log = logger.get(var='SERVING_LOGLEVEL', name='serving')


def start_model(opts, **kwargs):
    m = Model(name=opts['model_name'], **opts['model_params'], **kwargs)
    if opts.get('try_load', True):
        try:
            connector = S3InMemoryConnector(bucket=BUCKET, prefix=MODEL_PREFIX)
            m.load(connector=connector)
        except Exception as e:
            msg = """Did not load a state, will run with default initialized state.
                       Reason: {}"""
            log.warning(msg.format(e))
    else:
        log.warning("Config file specified that model does not load saved state, make sure this is correct")
    return m


def parse_data(data):
    parsed = {}
    try:
        parsed = json.loads(data[-1])
        parsed = dict([(k, v) if len(v) > 1 else (k, v[0]) for k, v in parsed.items()])
    except Exception as e:
        log.error("{}\tdata: {}".format(e, data))
    return parsed


def run_server(model, listener):
    """
    The server expects a \t separated byte string of (some, legacy, stuff, json with an actual data)
    and returns a \t separated list of the model name(+version) used, and a json with predictions.

    Parameters
    ----------
    model
        model to use for predictions

    listener
        an object that receives messages with data and sends replies with predictions
    """

    # Write file that can be used by Marathon to detect the service as ready to
    # receive requests.
    with open('/tmp/marathon-healthy', 'w') as f:
        f.write('1')

    while True:
        # receive new message
        msg = listener.recv()
        log.info("got message {}".format(msg))

        # split message
        data = msg.decode('ascii').split('\t')

        # parse message list
        parsed = parse_data(data)

        # use model to predict
        t_start = time.time()
        predictions = model.transform(parsed)
        t_end = time.time()

        # assemble reply
        results = [model.name, json.dumps(predictions)]

        # send reply
        listener.respond(bytes('\t'.join(results), 'ascii'))
        log.info("sent back {}".format(results))
        log.info("model time {}".format(t_end - t_start))


if __name__ == '__main__':
    # Parse args and modify CONFIG if needed, or maybe even load different
    # config depending on args.
    opts = docopt(__doc__)
    log.info(opts)

    # load config of model and data
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

    if opts['--no-load']:
        CONFIG['try_load'] = False

    model = start_model(CONFIG, rec_version=CONFIG['model_name'], log=log)

    if opts['--demo']:

        # would be cool to have some demonstration
        import numpy as np

        # this incoming data consists of
        # - random user id between 0 and 99 - random section-slug

        events = [
            # Make a dummy event...
            (
                '',
                json.dumps({
                    'user_id': str(user),
                    'url': 'http://www.latimes.com/{}/{}-story.html'.format(
                        *list(str(section_slug)))})
            )
            # .. 100 times.
            for user, section_slug in zip(
                np.random.randint(0, 100, 100),
                np.random.randint(10, 30, 100))
        ]  # End big list comprehension.

        listener = Demo(events)
    else:
        listener = Zmq(opts['--port'])

    run_server(model, listener)
