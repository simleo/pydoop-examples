"""\
Compare results from different models.

Input: for each model, *one* arrayblob containing all predictions.
"""

from itertools import combinations

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

import pydeep.arrayblob as arrayblob
import pydeep.common as common

LOGGER = logging.getLogger("compare_models")
WORKER = "cmpworker"
PACKAGE = "pydeep"


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-maps', metavar='INT', type=int, default=2)
    add_parser_common_arguments(parser)
    add_parser_arguments(parser)
    return parser


def generate_input_splits(N, input_dir, splits_path):
    """\
    Generate all possible blob pairs and assign a subset to each split.
    """
    blobs = [hdfs.path.join(input_dir, _)
             for _ in arrayblob.map_blobs(input_dir)]
    n_blobs = len(blobs)
    LOGGER.debug("%r: found %d blobs", input_dir, n_blobs)
    if n_blobs < 2:
        raise ValueError("not enough blobs for a comparison")
    n_pairs = n_blobs * (n_blobs - 1) // 2
    if N > n_pairs:
        N = n_pairs
        LOGGER.warn("Not enough blob pairs, will only do %d splits", N)
    splits = common.balanced_split(list(combinations(blobs, 2)), N)
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

    LOGGER.setLevel(args.log_level)

    splits_path = "%s_splits_%s" % (args.input, uuid.uuid4().hex)
    generate_input_splits(args.num_maps, args.input, splits_path)
    submitter = PydoopSubmitter()
    submitter.set_args(args, [] if unknown_args is None else unknown_args)
    submitter.properties.update({
        common.LOG_LEVEL_KEY: args.log_level,
        common.NUM_MAPS_KEY: args.num_maps,
        common.PYDOOP_EXTERNALSPLITS_URI_KEY: splits_path,
    })
    submitter.run()
    hdfs.rmr(splits_path)
    shutil.rmtree(wd)


if __name__ == "__main__":
    main(sys.argv)
