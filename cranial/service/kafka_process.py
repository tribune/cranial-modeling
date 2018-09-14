"""
Standard script for serving recommenders that use cranial.

Usage:
    kafka_process.py [--topic=TOPIC] [--config=<s>] [--model_suffix=<s>] [--restart_all_every_n=<i>] [--group=<s>]

Options:
--topic=TOPIC               Topic for Kafka listener. If not given the topic name will be model_name taken from config
--config=<s>                use a different config file.
                            Can also use a different config file stored in s3, use standard format s3://BUCKET/KEY
                            WARNING: if you decide to deploy models that differ from each
                            other by config only, make sure you also set different model names
--model_suffix=<s>          same config but modify model name by adding a suffix
--restart_all_every_n=<i>   if given, model, consumer and listener will be re-initialized after processing n messages
                            WARNING: Do not use this with small n (less than 100k), it might hurt performance
--group=<s>                 kafka consumer group, if not given defaults to model name
"""
# Core modules.
import json
import traceback
import os

# 3rd-party modules.
from docopt import docopt

# 1st-party modules.
from cranial.fetchers import S3InMemoryConnector
from cranial.fetchers.s3 import from_s3

# these are supposed to exist at the target location
from cranial.common import logger
from cranial.listeners.kafka import Listener as Kafka
from model import Model, Dataset, Consumer, BUCKET, MODEL_PREFIX

log = logger.get(name='kafka_process', var='MODELS_LOGLEVEL')


def run(opts):
    """ The server expects a \t separated ascii byte string of
          (user id, slug, yaml dict of additional parameters)
        and returns a \t separated list starting with the recommender algo
        version used, followed by a list of recommended slugs.

        @see parse_msg_list()
    """
    # make a var for convenience
    restart_every = opts['restart_all_every_n']
    restart_every = None if restart_every is None else int(restart_every)

    # init model, consumer and listener for the first time
    model, consumer, listener = init_objects(opts)

    # Write file that can be used by Marathon to detect the service as ready to
    # receive requests.
    with open('/tmp/marathon-healthy', 'w') as f:
        f.write('1')

    # start counter of processed messages
    c = 0
    while True:
        # receive new message
        msg = listener.recv(30)
        if (msg is not None) and len(msg):
            new_key = msg.decode('ascii')
            log.info("Got {}".format(new_key))
            process_key(model=model, consumer=consumer, opts=opts, s3_key=new_key)
            c += 1
            log.info("Finished processing {}".format(new_key))
            log.info("Total keys processed: {}".format(c))
        else:
            log.info("Got empty message, skipping...")

        if restart_every is not None and c > 0 and c % restart_every == 0:
            listener.consumer.close()
            model, consumer, listener = init_objects(opts)
            log.info("RESTARTED OBJECTS")


def init_objects(opts):
    """
    helper function to start and maybe download new model, start consumer and listener
    Parameters
    ----------
    opts
        opts from config file

    Returns
    -------
        model, consumer, listener
    """
    m = Model(name=opts['model_name'], **opts['model_params'])
    connector = S3InMemoryConnector(bucket=BUCKET, prefix=MODEL_PREFIX)
    m.load(connector=connector)
    log.info("started model {}".format(m))

    # create consumer
    c = Consumer(name=opts['model_name'], **opts['output_data_params'])
    log.info("started consumer {}".format(c))

    # create kafka listener
    l = Kafka(topic=opts['topic'], group=opts['group'])
    log.info("started listener {}".format(l))

    return m, c, l


def process_key(model, consumer, opts, s3_key: str) -> None:
    """
    This is the function is run by app.py which is a message consumer, message is send from
    content ingest job and contains a new s3 key.

    Download s3 key, vectorize new content, upload vectors to scylla

    Parameters
    ----------
    s3_key
        received s3 key
    """
    try:
        opts['input_data_params']['no_parallel'] = True
        opts['input_data_params']['keys_list'] = s3_key
        records = Dataset(**opts['input_data_params']).get_data()
        outputs = model.itransform(records)
        consumer.consume(outputs)
    except:
        log.error("Something failed...")
        log.error(traceback.format_exc())


if __name__ == '__main__':
    # Parse opts and modify CONFIG if needed, or maybe even load different
    # config depending on opts.
    opts = docopt(__doc__)

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
    log.info("final model name: {}".format(CONFIG['model_name']))

    CONFIG['topic'] = opts['--topic'] if opts['--topic'] is not None else CONFIG['topic']
    log.info("will listen to topic: {}".format(CONFIG['topic']))

    CONFIG['group'] = opts['--group'] if opts['--group'] is not None else CONFIG['model_name']
    log.info("will use group: {}".format(CONFIG['group']))

    CONFIG['restart_all_every_n'] = opts['--restart_all_every_n'] \
        if opts['--restart_all_every_n'] is not None else CONFIG.get('restart_all_every_n')
    log.info("will re-init objects every {} messages".format(CONFIG['restart_all_every_n']))

    # run everything
    run(opts=CONFIG)
