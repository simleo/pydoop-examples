"""\
Retrain the network with bottlenecks from the previous stage.

This application does not parallelize a single training on the whole image
dataset. Rather, it splits the dataset into subsets and uses each split
(training + validation) to get an independently retrained version of the
model. Repeating this with different number of splits allows to analyze
convergenge w.r.t. dataset size.
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import uuid
from hashlib import md5
from operator import itemgetter

from pydoop.app.submit import (
    add_parser_common_arguments, add_parser_arguments, PydoopSubmitter
)
from pydoop.utils.serialize import OpaqueInputSplit, write_opaques
from pydoop import hdfs

import pydeep.common as common
import pydeep.models as models

LOGGER = logging.getLogger("retrain_subsets")
WORKER = "rsworker"
PACKAGE = "pydeep"


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', metavar='STR',
                        default=models.DEFAULT)
    parser.add_argument('--batch-size', metavar='INT', type=int, default=100)
    parser.add_argument('--eval-step-interval', metavar='INT', type=int,
                        default=10)
    parser.add_argument('--learning-rate', metavar='FLOAT', type=float,
                        default=0.01)
    parser.add_argument('--num-maps', metavar='INT', type=int, default=4)
    parser.add_argument('--num-steps', metavar='INT', type=int, default=4000)
    parser.add_argument('--validation-percent', type=int, default=10)
    add_parser_common_arguments(parser)
    add_parser_arguments(parser)
    return parser


def map_bnecks_to_files(input_dir, record_size):
    """\
    For each input subdir (corresponding to an image class), build the full
    list of (filename, offset) pair where each bottleneck dump can be
    retrieved.

    {'hdfs://.../bottlenecks/dandelion': [
        ('part-m-00000', 0),
        ('part-m-00000', 8192),
        ...
        ('part-m-00003', 163840)
    ],
    'hdfs://.../bottlenecks/roses': [
        ('part-m-00000', 0),
        ...
    ]}
    """
    m = {}
    with hdfs.hdfs() as fs:
        for stat in fs.list_directory(input_dir):
            if stat['kind'] != 'directory':
                continue
            subd = stat['name']
            stats = [_ for _ in fs.list_directory(subd)
                     if not hdfs.path.basename(_["name"]).startswith("_")]
            stats.sort(key=itemgetter("name"))
            m[subd] = stats
    for subd, stats in m.items():
        positions = []
        for s in stats:
            bname = hdfs.path.basename(s["name"])
            assert s["size"] % record_size == 0
            for i in range(0, s["size"], record_size):
                positions.append((bname, i))
        m[subd] = positions
    return m


def generate_input_splits(N, bneck_map, splits_path):
    """\
    Assign to each split a chunk of bottlenecks across all classes.
    """
    bneck_map = {d: list(common.balanced_split(locs, N))
                 for d, locs in bneck_map.items()}
    splits = [{d: seq[i] for d, seq in bneck_map.items()} for i in range(N)]
    LOGGER.debug("saving input splits to: %s", splits_path)
    with hdfs.open(splits_path, 'wb') as f:
        write_opaques([OpaqueInputSplit(1, _) for _ in splits], f)


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
    args.num_reducers = 0

    model = models.get_model_info(args.architecture)
    graph = model.load_prep()
    bneck_tensor = model.get_bottleneck(graph)
    bneck_size = bneck_tensor.dtype.size * bneck_tensor.shape[1].value
    record_size = md5().digest_size + bneck_size
    bneck_map = map_bnecks_to_files(args.input, record_size)
    LOGGER.info("%d subdirs, %r bottlenecks" %
                (len(bneck_map), [len(_) for _ in bneck_map.values()]))
    splits_path = os.path.join(args.input, '_' + uuid.uuid4().hex)
    generate_input_splits(args.num_maps, bneck_map, splits_path)
    submitter = PydoopSubmitter()
    submitter.set_args(args, [] if unknown_args is None else unknown_args)
    submitter.properties.update({
        common.EVAL_STEP_INTERVAL_KEY: args.eval_step_interval,
        common.GRAPH_ARCH_KEY: args.architecture,
        common.LEARNING_RATE_KEY: args.learning_rate,
        common.LOG_LEVEL_KEY: args.log_level,
        common.NUM_CLASSES_KEY: len(bneck_map),
        common.NUM_MAPS_KEY: args.num_maps,
        common.NUM_STEPS_KEY: args.num_steps,
        common.PYDOOP_EXTERNALSPLITS_URI_KEY: splits_path,
        common.TRAIN_BATCH_SIZE_KEY: args.batch_size,
        common.VALIDATION_PERCENT_KEY: args.validation_percent,
    })
    submitter.run()
    hdfs.rmr(splits_path)
    shutil.rmtree(wd)


if __name__ == "__main__":
    main(sys.argv)
