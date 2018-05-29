"""\
Calculate all image feature vectors (bottlenecks) for the given
network architecture.

Input
-----

HDFS dir with one subdir per class, each containing JPEG images of items
belonging to that class. At least two classes are required. Image files must
end in .jpg or .jpeg.

flower_photos/
|-- roses
|   |-- bar.jpg
|   `-- foo.jpg
`-- tulips
    |-- tar.jpg
    |-- taz.jpg
    `-- waz.jpg

Output
------

HDFS dir with one subdir per class, each containing one part* file for each
map task. Each file contains a raw binary dump of all feature vectors, as
returned by numpy.ndarray.tobytes, one right after the other.

bottlenecks/
|-- roses
|   |-- part-m-00000
|   |-- part-m-00001
|   `-- _SUCCESS
`-- tulips
    |-- part-m-00000
    |-- part-m-00001
    `-- _SUCCESS

One separate Hadoop job is submitted for each class.
"""

from copy import deepcopy
from threading import Thread
from queue import Queue
import argparse
import logging
import os
import shutil
import sys
import tempfile
import uuid

from pydoop.app.submit import (
    add_parser_common_arguments, add_parser_arguments, PydoopSubmitter
)
from pydoop.utils.serialize import OpaqueInputSplit, write_opaques
from pydoop import hdfs

from pydeep.tflow import BottleneckProjector, get_model_graph, save_graph
from pydeep.models import model
from pydeep.keys import GRAPH_PATH_KEY, GRAPH_ARCH_KEY


LOGGER = logging.getLogger("genbnecks")
RETVALS = Queue()
PACKAGE = "pydeep"

DEFAULT_NUM_MAPS = 10
DEFAULT_ARCHITECTURE = 'inception_v3'
NUM_MAPS_KEY = 'mapreduce.job.maps'
# FIXME this should be from pydoop.utils.serialize
PYDOOP_EXTERNALSPLITS_URI_KEY = 'pydoop.mapreduce.pipes.externalsplits.uri'


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-maps', metavar='INT', type=int, default=DEFAULT_NUM_MAPS,
    )
    parser.add_argument(
        '--architecture', metavar='STR', default=DEFAULT_ARCHITECTURE,
    )
    add_parser_common_arguments(parser)
    add_parser_arguments(parser)
    return parser


def map_input_files(input_dir):
    """\
    Map each class to the list of available files for that class.
    """
    img_map = {}
    ext = frozenset(('jpg', 'jpeg'))
    with hdfs.hdfs() as fs:
        for stat in fs.list_directory(input_dir):
            if stat['kind'] == 'directory':
                cls = stat['name'].rsplit('/', 1)[-1]
                img_map[cls] = [
                    _['name'] for _ in fs.list_directory(stat['name'])
                    if _['name'].rsplit('.', 1)[-1].lower() in ext
                ]
    return img_map


def add_D_arg(args, arg_name, arg_key):
    val = str(getattr(args, arg_name))
    if args.D is None:
        args.D = [[arg_key, val]]
    elif not any(map(lambda _: _[0] == arg_key, args.D)):
        args.D.append([arg_key, val])


def prepare_and_save_graph(model, graph_path):
    if not hdfs.path.exists(model['path']):
        get_model_graph(model)
    LOGGER.info("creating graph")
    graph = BottleneckProjector.create_graph(model)
    LOGGER.info("saving graph to %s", graph_path)
    save_graph(graph, graph_path)


def generate_input_splits(uri, n_mappers, images):
    groups = [[] for _ in range(n_mappers)]
    for i, img in enumerate(images):
        groups[i % n_mappers].append(img)
    with hdfs.open(uri, 'wb') as f:
        write_opaques([OpaqueInputSplit(1, _) for _ in groups], f)


def run_map_job(args, unknown_args, images):
    logger = logging.getLogger(args.job_name)
    logger.setLevel(args.log_level)
    uri = os.path.join(args.input, '_' + uuid.uuid4().hex)
    logger.debug("saving input splits to: %s", uri)
    generate_input_splits(uri, args.num_maps, images)
    args.D.append([PYDOOP_EXTERNALSPLITS_URI_KEY, uri])
    submitter = PydoopSubmitter()
    submitter.set_args(args, [] if unknown_args is None else unknown_args)
    submitter.run()
    hdfs.rmr(uri)
    RETVALS.put_nowait(0)


def main(argv=None):

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    wd = tempfile.mkdtemp(prefix="pydeep_")
    zip_fn = os.path.join(wd, "{}.zip".format(PACKAGE))
    shutil.make_archive(*zip_fn.rsplit(".", 1), base_dir=PACKAGE)

    parser = make_parser()
    args, unknown_args = parser.parse_known_args(argv)
    args.job_name = 'genbnecks'
    args.module = 'bworker'
    args.upload_file_to_cache = ['bworker.py']
    args.python_zip = [zip_fn]
    args.do_not_use_java_record_reader = True
    args.do_not_use_java_record_writer = True

    try:
        m = model[args.architecture]
    except KeyError:
        sys.exit("ERROR: unknown architecture: {}".format(args.architecture))

    LOGGER.setLevel(args.log_level)
    img_map = map_input_files(args.input)
    LOGGER.info("%d classes, %d total images",
                len(img_map), sum(map(len, img_map.values())))
    graph_path = 'graph-{}.pb'.format(uuid.uuid4().hex)
    prepare_and_save_graph(m, graph_path)
    add_D_arg(args, 'num_maps', NUM_MAPS_KEY)
    add_D_arg(args, 'architecture', GRAPH_ARCH_KEY)
    args.D.append([GRAPH_PATH_KEY, graph_path])
    args.num_reducers = 0

    hdfs.mkdir(args.output)
    threads = []
    for cls, img_list in img_map.items():
        nargs = deepcopy(args)
        nargs.job_name += '-' + cls
        nargs.output = os.path.join(nargs.output, cls)
        t = Thread(target=run_map_job, args=[nargs, unknown_args, img_list])
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    shutil.rmtree(wd)
    if RETVALS.qsize() < len(threads):
        sys.exit("ERROR: one or more workers failed")


if __name__ == "__main__":
    main(sys.argv)
