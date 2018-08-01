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
import random
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
import pydeep.ioformats as ioformats

LOGGER = logging.getLogger("retrain_subsets")
WORKER = "rsworker"
PACKAGE = "pydeep"


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', metavar='STR',
                        default=models.DEFAULT)
    parser.add_argument('--train-batch-size', metavar='INT', type=int,
                        default=100)
    parser.add_argument('--validation-batch-size', metavar='INT', type=int,
                        default=100)
    parser.add_argument('--eval-step-interval', metavar='INT', type=int,
                        default=10)
    parser.add_argument('--learning-rate', metavar='FLOAT', type=float,
                        default=0.01)
    parser.add_argument('--num-maps', metavar='INT', type=int, default=4)
    parser.add_argument('--num-steps', metavar='INT', type=int, default=4000)
    parser.add_argument('--seed', metavar='INT', type=int)
    parser.add_argument('--validation-percent', type=int, default=10)
    add_parser_common_arguments(parser)
    add_parser_arguments(parser)
    return parser


def generate_input_splits(N, bneck_map, splits_path):
    """\
    Assign to each split a chunk of bottlenecks across all classes.
    """
    for locs in bneck_map.values():
        random.shuffle(locs)
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
    if args.seed:
        LOGGER.info("setting random seed to %d", args.seed)
        random.seed(args.seed)

    model = models.get_model_info(args.architecture)
    graph = model.load_prep()
    bneck_tensor = model.get_bottleneck(graph)
    bneck_store = ioformats.BottleneckStore(
        args.input, bneck_tensor.shape[1].value, bneck_tensor.dtype
    )
    bneck_map = bneck_store.posmap
    LOGGER.info("%d subdirs, %r bottlenecks" %
                (len(bneck_map), [len(_) for _ in bneck_map.values()]))
    splits_path = os.path.join(args.input, '_' + uuid.uuid4().hex)
    generate_input_splits(args.num_maps, bneck_map, splits_path)
    submitter = PydoopSubmitter()
    submitter.set_args(args, [] if unknown_args is None else unknown_args)
    submitter.properties.update({
        common.BNECKS_DIR_KEY: args.input,
        common.EVAL_STEP_INTERVAL_KEY: args.eval_step_interval,
        common.GRAPH_ARCH_KEY: args.architecture,
        common.LEARNING_RATE_KEY: args.learning_rate,
        common.LOG_LEVEL_KEY: args.log_level,
        common.NUM_MAPS_KEY: args.num_maps,
        common.NUM_STEPS_KEY: args.num_steps,
        common.PYDOOP_EXTERNALSPLITS_URI_KEY: splits_path,
        common.TRAIN_BATCH_SIZE_KEY: args.train_batch_size,
        common.VALIDATION_BATCH_SIZE_KEY: args.validation_batch_size,
        common.VALIDATION_PERCENT_KEY: args.validation_percent,
    })
    if args.seed:
        submitter.properties[common.SEED_KEY] = args.seed
    submitter.run()
    hdfs.rmr(splits_path)
    shutil.rmtree(wd)


if __name__ == "__main__":
    main(sys.argv)
