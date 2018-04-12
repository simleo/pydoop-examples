"""
This example is a more-or-less direct map-reduce reinterpretation
of the [tensorflow for poets](FIXME-link) example.
"""

from pydoop.app.submit import (
    add_parser_common_arguments,
    add_parser_arguments)
from pydoop.app.submit import PydoopSubmitter
from pydoop.utils.serialize import OpaqueInputSplit, write_opaques
from pydoop import hdfs

from tflow import BottleneckProjector, save_graph
from models import model
from keys import GRAPH_PATH_KEY, GRAPH_ARCH_KEY

from multiprocessing import Process
from copy import deepcopy
import sys
import itertools as it
import argparse
import random
import uuid
import os

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


def get_categories_data(path):
    categories = {}
    for x in os.scandir(path):
        if x.is_dir():
            categories[x.name] = [os.path.join(x.path, f)
                                  for f in os.listdir(x.path)
                                  if (f.split('.')[-1].lower()
                                      in ['jpg', 'jpeg'])]
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
    graph = BottleneckProjector.create_graph(model)
    save_graph(graph, graph_path)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return it.zip_longest(*args, fillvalue=fillvalue)


def generate_input_splits(uri, n_mappers, images):
    n = len(images) // n_mappers
    opaques = [OpaqueInputSplit(1, list(g)) for g in grouper(images, n)]
    write_opaques(opaques, uri)


def run_map_job(args, unknown_args, images):
    uri = os.path.join(args.input, '__' + uuid.uuid4().hex)
    generate_input_splits(uri, images)
    args.D.append([PYDOOP_EXTERNALSPLITS_URI_KEY, uri])
    submitter = PydoopSubmitter()
    submitter.set_args(args, [] if unknown_args is None else unknown_args)
    submitter.run()


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

    hdfs.mkdir(args.input)  # FIXME check if it is already there
    categories = get_categories_data(args.data_dir)
    random_shuffle(categories)
    graph_path = os.path.join(args.input,
                              '__graph-' + uuid.uuid4().hex + '.pb')
    prepare_and_save_graph(model[args.architecture], graph_path)
    add_D_arg(args, 'num_maps', NUM_MAPS_KEY)
    add_D_arg(args, 'architecture', GRAPH_ARCH_KEY)
    args.D.append([GRAPH_PATH_KEY, graph_path])
    args.num_reducers = 0

    hdfs.mkdir(args.output)  # FIXME check if it is already there
    # multi-thread on the following
    procs = []
    for name in categories:
        nargs = deepcopy(args)
        nargs.job_name += '-' + name
        nargs.output = os.path.join(nargs.output, name)
        p = Process(target=run_map_job,
                    args=[nargs, unknown_args, categories[name]])
        p.append(p)
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main(sys.argv)
