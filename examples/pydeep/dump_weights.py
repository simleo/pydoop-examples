"""\
Dump model weights (and biases).
"""

import argparse
import logging
import os
import re
import sys

import numpy as np
from pydoop import hdfs
import tensorflow as tf

import pydeep.models as models
from pydeep.common import LOG_LEVELS

logging.basicConfig()
LOGGER = logging.getLogger("dump_weights")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", metavar="INPUT_DIR")
    parser.add_argument("--architecture", metavar="STR",
                        default=models.DEFAULT)
    parser.add_argument("--out-weights", metavar="PATH")
    parser.add_argument("--out-biases", metavar="PATH")
    parser.add_argument("--log-level", metavar="|".join(LOG_LEVELS),
                        choices=LOG_LEVELS, default="INFO")
    return parser


def get_wb(model, path):
    with tf.Session(graph=tf.Graph()) as session:
        models.load_checkpoint(path)
        graph = session.graph
        weights, biases = graph.get_collection("trainable_variables")
        if len(weights.shape) == 1:
            assert len(biases.shape) == 2
            weights, biases = biases, weights
        return session.run((weights, biases))


def main(argv=sys.argv):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = make_parser()
    args = parser.parse_args(argv[1:])
    LOGGER.setLevel(args.log_level)
    if args.out_weights is None:
        args.out_weights = hdfs.path.join("%s_weights.npz" % args.input)
    if args.out_biases is None:
        args.out_biases = hdfs.path.join("%s_biases.npz" % args.input)
    paths = []
    tags = {}
    for p in hdfs.ls(args.input):
        m = re.match(r"^part-m-(\d+)\.zip$", hdfs.path.basename(p))
        if m:
            paths.append(p)
            tags[p] = m.groups()[0]
    model = models.get_model_info(args.architecture)
    weights, biases = {}, {}
    for p in paths:
        t = tags[p]
        weights[t], biases[t] = get_wb(model, p)
        LOGGER.info("%s: W %r b %r", p, weights[t].shape, biases[t].shape)
    with hdfs.open(args.out_weights, "wb") as f:
        np.savez(f, **weights)
    with hdfs.open(args.out_biases, "wb") as f:
        np.savez(f, **biases)


if __name__ == "__main__":
    sys.exit(main())
