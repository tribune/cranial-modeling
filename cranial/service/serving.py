"""
Standard script for serving prediction that use cranial.

Usage:
    serving.py [--demo] [--port=PORT] [--no-load] [--config=<s>] [--model_suffix=<s>] [--firehose=<s>]

Options:
--demo              run demo mode
--port=PORT         Port for ZMQ listener. [default: 5678]
--no-load           if set, model will not load from previously saved state
--config=<s>        use a different config file.
                    Can also use a different config file stored in s3, use standard format s3://BUCKET/KEY
                    WARNING: if you decide to deploy models that differ from each
                    other by config only, make sure you also set different model names
                    UNLESS different configs are for differentiating between updater and
                    predictor nodes, then model name should be the same
--model_suffix=<s>  same config but modify model name by adding a suffix
--firehose=<s>      override name of the firehose stream, defaults to what is provided in model config,
                    if not set there then it defaults to the model name with suffix
"""
# Core modules.
import json
import time
import os
import traceback
import arrow

# 3rd-party modules.
from docopt import docopt

# 1st-party modules.
from cranial.fetchers import S3InMemoryConnector
from cranial.fetchers.s3 import from_s3
from cranial.listeners.zmq import Listener as Zmq
from cranial.messaging.adapters import firehose_async
from cranial.common import logger

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

def set_ready():
    with open('/tmp/serving-healthy', 'w') as f:
        # Two minutes grace for start-up time.
        f.write(str(time.time() + 120.0 ))


def set_last_response_time(t: float):
    with open('/tmp/serving-healthy', 'w') as f:
        f.write(str(t))


def run_server(model, listener, firehose_name=None):
    """
    The server expects a \t separated ascii byte string of (some, legacy, stuff, json with an actual data)
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
    set_ready()

    while True:
        log.info("WAITING FOR MESSAGES FROM LISTENER.....")
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
        set_last_response_time(t_end)

        # assemble reply
        results = [model.name, json.dumps(predictions)]

        # send reply
        listener.resp(bytes('\t'.join(results), 'ascii'))
        log.info("sent back {}".format(results))
        log.info("model time {}".format(t_end - t_start))
        if firehose_name is not None:
            firehose_async.put_data(firehose_name,
                                    json.dumps({'time_iso8601': str(arrow.utcnow()),
                                                'model_response_time': t_end - t_start, **parsed, **predictions}))


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

    # try to set up firehose logging
    # firehose should be specified either in config file or by script option
    # if not specified in either place then there will be no logging
    firehose_name = CONFIG.get('firehose') if opts['--firehose'] is None else opts['--firehose']
    if firehose_name is not None:
        # this will throw an error if file not found
        with open('/keys/firehose-write.json') as creds:
            firehose_async.firehose.auth(json.load(creds))

    # sometimes it is useful to nod load a model but just instantiate
    if opts['--no-load']:
        CONFIG['try_load'] = False

    # start model
    model = start_model(CONFIG, rec_version=CONFIG['model_name'], log=log)

    listener = Zmq(opts['--port'])

    run_server(model, listener, firehose_name=firehose_name)
