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
"""

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
import uuid

from pydoop.app.submit import (
    add_parser_common_arguments, add_parser_arguments, PydoopSubmitter
)
from pydoop.utils.serialize import OpaqueInputSplit, write_opaques
from pydoop import hdfs

from pydeep.models import get_model_info, DEFAULT as DEFAULT_ARCHITECTURE
import pydeep.common as common

from graph_setup import get_graph


LOGGER = logging.getLogger("genbnecks")
WORKER = "bworker"
PACKAGE = "pydeep"

DEFAULT_NUM_MAPS = 10


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


def list_images(input_dir):
    ret = []
    p = re.compile(r".*\.jpe?g$", re.IGNORECASE)
    ls = [_['name'] for _ in hdfs.lsl(input_dir) if _['kind'] == 'directory']
    for d in ls:
        ret.extend([_ for _ in hdfs.ls(d) if p.match(_)])
    LOGGER.info("%d classes, %d total images", len(ls), len(ret))
    return ret


def main(argv=None):

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    wd = tempfile.mkdtemp(prefix="pydeep_")
    zip_fn = os.path.join(wd, "{}.zip".format(PACKAGE))
    shutil.make_archive(*zip_fn.rsplit(".", 1), base_dir=PACKAGE)

    parser = make_parser()
    args, unknown_args = parser.parse_known_args(argv)
    args.job_name = WORKER
    args.module = WORKER
    args.upload_file_to_cache = ['%s.py' % WORKER]
    args.python_zip = [zip_fn]
    args.do_not_use_java_record_reader = True
    args.do_not_use_java_record_writer = True
    args.num_reducers = 0

    LOGGER.setLevel(args.log_level)
    model = get_model_info(args.architecture)
    get_graph(model, log_level=args.log_level)

    images = list_images(args.input)
    splits = common.balanced_split(images, args.num_maps)
    uri = os.path.join(args.input, '_' + uuid.uuid4().hex)
    LOGGER.debug("saving input splits to: %s", uri)
    with hdfs.open(uri, 'wb') as f:
        write_opaques([OpaqueInputSplit(1, _) for _ in splits], f)
    submitter = PydoopSubmitter()
    submitter.set_args(args, [] if unknown_args is None else unknown_args)
    submitter.properties.update({
        common.NUM_MAPS_KEY: args.num_maps,
        common.GRAPH_ARCH_KEY: args.architecture,
        common.PYDOOP_EXTERNALSPLITS_URI_KEY: uri,
    })
    submitter.run()
    hdfs.rmr(uri)
    shutil.rmtree(wd)


if __name__ == "__main__":
    main(sys.argv)
