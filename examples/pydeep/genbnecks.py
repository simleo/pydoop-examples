"""\
Calculate all image feature vectors (bottlenecks) for the given
network architecture.
"""

from copy import deepcopy
from threading import Thread
from queue import Queue
import argparse
import itertools as it
import logging
import os
import random
import sys
import uuid

from pydoop.app.submit import (
    add_parser_common_arguments, add_parser_arguments, PydoopSubmitter
)
from pydoop.utils.serialize import OpaqueInputSplit, write_opaques
from pydoop import hdfs

from tflow import BottleneckProjector, get_model_graph, save_graph
from models import model
from keys import GRAPH_PATH_KEY, GRAPH_ARCH_KEY


LOGGER = logging.getLogger("genbnecks")
RETVALS = Queue()

# Pre-assembled options
DEFAULT_NUM_MAPS = 10
DEFAULT_ARCHITECTURE = 'inception_v3'
NUM_MAPS_KEY = 'mapreduce.job.maps'
# FIXME this should be from pydoop.utils.serialize
PYDOOP_EXTERNALSPLITS_URI_KEY = 'pydoop.mapreduce.pipes.externalsplits.uri'


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-maps', metavar='INT', type=int,
        default=DEFAULT_NUM_MAPS,
        help=''
    )
    parser.add_argument(
        '--architecture', metavar='STR', type=str,
        default=DEFAULT_ARCHITECTURE,
        help=''
    )
    add_parser_common_arguments(parser)
    add_parser_arguments(parser)
    return parser


def get_categories_data(input_dir):
    categories = {}
    ext = frozenset(('jpg', 'jpeg'))
    with hdfs.hdfs() as fs:
        for stat in fs.list_directory(input_dir):
            if stat['kind'] == 'directory':
                cat = stat['name'].rsplit('/', 1)[-1]
                categories[cat] = [
                    _['name'] for _ in fs.list_directory(stat['name'])
                    if _['name'].rsplit('.', 1)[-1].lower() in ext
                ]
    return categories


def random_shuffle(categories):
    for k in categories:
        random.shuffle(categories[k])


def add_D_arg(args, arg_name, arg_key):
    val = str(getattr(args, arg_name))
    if args.D is None:
        args.D = [[arg_key, val]]
    elif not any(map(lambda _: _[0] == arg_key,
                 args.D)):
        args.D.append([arg_key, val])


def prepare_and_save_graph(model, graph_path):
    if not hdfs.path.exists(model['path']):
        get_model_graph(model)
    LOGGER.info("creating graph")
    graph = BottleneckProjector.create_graph(model)
    LOGGER.info("saving graph to %s", graph_path)
    save_graph(graph, graph_path)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return it.zip_longest(*args, fillvalue=fillvalue)


def generate_input_splits(uri, n_mappers, images):
    n = len(images) // n_mappers
    opaques = [OpaqueInputSplit(1, list(g)) for g in grouper(images, n)]
    with hdfs.open(uri, 'wb') as f:
        write_opaques(opaques, f)


def run_map_job(args, unknown_args, images):
    logger = logging.getLogger(args.job_name)
    logger.setLevel(args.log_level)
    uri = os.path.join(args.input, '__' + uuid.uuid4().hex)
    logger.debug("saving input splits to: %s", uri)
    generate_input_splits(uri, args.num_maps, images)
    args.D.append([PYDOOP_EXTERNALSPLITS_URI_KEY, uri])
    submitter = PydoopSubmitter()
    submitter.set_args(args, [] if unknown_args is None else unknown_args)
    submitter.run()
    RETVALS.put_nowait(0)


def main(argv=None):
    parser = make_parser()
    args, unknown_args = parser.parse_known_args(argv)
    args.job_name = 'bworker'
    args.module = 'bworker'
    args.upload_file_to_cache = [
        'models.py', 'bworker.py', 'tflow.py', 'ioformats.py']
    args.python_zip = ['']
    args.do_not_use_java_record_reader = True
    args.do_not_use_java_record_writer = True

    try:
        m = model[args.architecture]
    except KeyError:
        sys.exit("ERROR: unknown architecture: {}".format(args.architecture))

    LOGGER.setLevel(args.log_level)
    categories = get_categories_data(args.input)
    LOGGER.info("%d categories, %d total images",
                len(categories), sum(map(len, categories.values())))
    random_shuffle(categories)
    graph_path = 'graph-{}.pb'.format(uuid.uuid4().hex)
    prepare_and_save_graph(m, graph_path)
    add_D_arg(args, 'num_maps', NUM_MAPS_KEY)
    add_D_arg(args, 'architecture', GRAPH_ARCH_KEY)
    args.D.append([GRAPH_PATH_KEY, graph_path])
    args.num_reducers = 0

    hdfs.mkdir(args.output)
    procs = []
    for name in categories:
        nargs = deepcopy(args)
        nargs.job_name += '-' + name
        nargs.output = os.path.join(nargs.output, name)
        p = Thread(target=run_map_job,
                   args=[nargs, unknown_args, categories[name]])
        procs.append(p)
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    if RETVALS.qsize() < len(procs):
        sys.exit("ERROR: one or more workers failed")


if __name__ == "__main__":
    main(sys.argv)
