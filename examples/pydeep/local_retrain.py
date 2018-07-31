"""\
Local (no MapReduce) Reimplementation of the original tf-for-poets example.

All paths are assumed to be HDFS paths by default.
"""

import argparse
from hashlib import md5
import logging
import os
import random
import sys

from pydoop import hdfs

import pydeep.models as models
import pydeep.tflow as tflow
from pydeep.ioformats import BottleneckStore
from pydeep.common import LOG_LEVELS

from graph_setup import get_graph
from genbnecks import list_images

logging.basicConfig()
LOGGER = logging.getLogger("local_retrain")

BNECKS_BASENAME = "part-m-00000"
DEFAULT_TRAINED_MODEL = "trained_model.zip"
DEFAULT_TRAIN_OUTPUT = "training.tsv"


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", metavar="INPUT_DIR")
    parser.add_argument("--architecture", metavar="STR",
                        default=models.DEFAULT)
    parser.add_argument('--bnecks-dir', metavar='DIR', default='bottlenecks')
    parser.add_argument('--eval-step-interval', metavar='INT', type=int,
                        default=10)
    parser.add_argument('--learning-rate', metavar='FLOAT', type=float,
                        default=0.01)
    parser.add_argument('--num-steps', metavar='INT', type=int, default=4000)
    parser.add_argument('--test-batch-size', metavar='INT', type=int,
                        default=-1)
    parser.add_argument('--test-percent', type=int, default=10)
    parser.add_argument('--train-batch-size', metavar='INT', type=int,
                        default=100)
    parser.add_argument("--train-output", metavar="FILE",
                        default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--trained-model", metavar="FILE",
                        default=DEFAULT_TRAINED_MODEL)
    parser.add_argument('--validation-batch-size', metavar='INT', type=int,
                        default=100)
    parser.add_argument('--validation-percent', type=int, default=10)
    parser.add_argument("--log-level", metavar="|".join(LOG_LEVELS),
                        choices=LOG_LEVELS, default="INFO")
    return parser


def map_input_files(input_dir):
    ret = {}
    for path in list_images(input_dir):
        cls = path.rsplit("/", 2)[1]
        ret.setdefault(cls, []).append(path)
    return ret


def calc_bottlenecks(model, img_map, out_dir):
    projector = tflow.BottleneckProjector(model)
    for in_subd, img_paths in img_map.items():
        cls = hdfs.path.basename(in_subd)
        out_subd = hdfs.path.join(out_dir, cls)
        hdfs.mkdir(out_subd)
        bnecks_path = hdfs.path.join(out_subd, BNECKS_BASENAME)
        LOGGER.info("computing bottlenecks for: %s", cls)
        with hdfs.open(bnecks_path, "wb") as out_f:
            for path in img_paths:
                with hdfs.open(path, "rb") as in_f:
                    data = in_f.read()
                    checksum = md5(data).digest()
                    bneck = projector.project(data)
                    out_f.write(checksum + bneck.tobytes())
    projector.close_session()


def get_bneck_maps(model, bnecks_dir, test_percent, val_percent):
    graph = model.load_prep()
    bneck_tensor = model.get_bottleneck(graph)
    bneck_store = BottleneckStore(
        bneck_tensor.shape[1].value, bneck_tensor.dtype
    )
    del graph
    bneck_map = bneck_store.get_bnecks(bnecks_dir)
    for bnecks in bneck_map.values():
        random.shuffle(bnecks)
    test_bneck_map = {c: [] for c in bneck_map}
    val_bneck_map = {c: [] for c in bneck_map}
    for c, bnecks in bneck_map.items():
        test_n = round(test_percent * len(bnecks) / 100)
        val_n = round(val_percent * len(bnecks) / 100)
        for _ in range(test_n):
            test_bneck_map[c].append(bnecks.pop())
        for _ in range(val_n):
            val_bneck_map[c].append(bnecks.pop())
    return bneck_map, val_bneck_map, test_bneck_map


def get_sample_vectors(bneck_map, batch_size, labels):
    bs_map = {}
    for c, bnecks in bneck_map.items():
        capped = min(batch_size, len(bnecks))
        bs_map[c] = capped if capped > 0 else len(bnecks)
    batch = {c: random.sample(bnecks, bs_map[c])
             for c, bnecks in bneck_map.items()}
    return BottleneckStore.bnecks_map_to_vectors(batch, labels)


def dump_stats(f, i, cross_entropy, train_acc, val_acc):
    LOGGER.info(
        "step %d: cross entropy = %f, train acc. = %f%%, val acc. = %f%%",
        i, cross_entropy, 100 * train_acc, 100 * val_acc
    )
    f.write("%d\t%s\t%s\t%s" % (i, cross_entropy, train_acc, val_acc))


def main(argv=sys.argv):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = make_parser()
    args = parser.parse_args(argv[1:])
    LOGGER.setLevel(args.log_level)
    model = models.get_model_info(args.architecture)
    get_graph(model, log_level=args.log_level)
    img_map = map_input_files(args.input)
    LOGGER.info("%d classes, %r images",
                len(img_map), [len(_) for _ in img_map.values()])
    if hdfs.path.exists(args.bnecks_dir):
        LOGGER.info("%r already exists, skipping bottleneck calculation",
                    args.bnecks_dir)
    else:
        LOGGER.info("caching bottlenecks to %r", args.bnecks_dir)
        calc_bottlenecks(model, img_map, args.bnecks_dir)
    train_bneck_map, val_bneck_map, test_bneck_map = maps = get_bneck_maps(
        model, args.bnecks_dir, args.test_percent, args.validation_percent
    )
    for cat, m in zip(("train", "val", "test"), maps):
        LOGGER.info("%s set: %r", cat, [len(_) for _ in m.values()])
    labels = BottleneckStore.assign_labels(args.bnecks_dir)
    n_classes = len(labels)
    for m in (img_map, *maps):
        assert len(m) == n_classes

    # train
    retrainer = tflow.Retrainer(model, n_classes, args.learning_rate)
    for i in range(args.num_steps):
        train_bnecks, train_gtruths = get_sample_vectors(
            train_bneck_map, args.train_batch_size, labels
        )
        val_bnecks, val_gtruths = get_sample_vectors(
            val_bneck_map, args.validation_batch_size, labels
        )
        retrainer.run_train_step(train_bnecks, train_gtruths)
        with hdfs.open(args.train_output, "wt") as f:
            if (i % args.eval_step_interval == 0) or (i + 1 >= args.num_steps):
                train_accuracy, cross_entropy = retrainer.run_eval_step(
                    train_bnecks, train_gtruths
                )
                val_accuracy = retrainer.run_validation_step(
                    val_bnecks, val_gtruths
                )
                dump_stats(f, i, cross_entropy, train_accuracy, val_accuracy)
    retrainer.checkpoint(args.trained_model)

    # test
    test_bnecks, test_gtruths = get_sample_vectors(
        test_bneck_map, args.test_batch_size, labels
    )
    test_accuracy, predictions = retrainer.session.run(
        [retrainer.eval_step, retrainer.final_tensor],
        feed_dict={retrainer.bneck_input: test_bnecks,
                   retrainer.ground_truth_input: test_gtruths})
    print("test_accuracy: %f%%" % (100 * test_accuracy))
    retrainer.close_session()
    # TODO: output predictions and/or misclassified images


if __name__ == "__main__":
    sys.exit(main())
