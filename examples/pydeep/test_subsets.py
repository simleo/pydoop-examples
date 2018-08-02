"""\
Test models trained with retrain_subsets.py on the whole dataset.
"""

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

import pydeep.common as common
import pydeep.models as models

LOGGER = logging.getLogger("test_subsets")
WORKER = "tsworker"
PACKAGE = "pydeep"


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', metavar='STR',
                        default=models.DEFAULT)
    parser.add_argument('--num-maps', metavar='INT', type=int, default=2)
    parser.add_argument('--bnecks-dir', metavar='DIR', default='bottlenecks')
    add_parser_common_arguments(parser)
    add_parser_arguments(parser)
    return parser


def generate_input_splits(N, input_dir, splits_path):
    """\
    Assign a subset of the trained models to each split.
    """
    paths = [_ for _ in hdfs.ls(input_dir) if
             hdfs.path.basename(_).endswith(".zip")]
    if N > len(paths):
        N = len(paths)
        LOGGER.warn("Not enough input models, will only do %d splits" % N)
    splits = common.balanced_split(paths, N)
    LOGGER.debug("saving input splits to: %s", splits_path)
    with hdfs.open(splits_path, 'wb') as f:
        write_opaques([OpaqueInputSplit(1, _) for _ in splits], f)
    return N


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

    splits_path = os.path.join(args.input, '_' + uuid.uuid4().hex)
    generate_input_splits(args.num_maps, args.input, splits_path)
    submitter = PydoopSubmitter()
    submitter.set_args(args, [] if unknown_args is None else unknown_args)
    submitter.properties.update({
        common.BNECKS_DIR_KEY: args.bnecks_dir,
        common.GRAPH_ARCH_KEY: args.architecture,
        common.LOG_LEVEL_KEY: args.log_level,
        common.NUM_MAPS_KEY: args.num_maps,
        common.PYDOOP_EXTERNALSPLITS_URI_KEY: splits_path,
    })
    submitter.run()
    hdfs.rmr(splits_path)
    shutil.rmtree(wd)


if __name__ == "__main__":
    main(sys.argv)
