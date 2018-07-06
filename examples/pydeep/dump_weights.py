"""\
Dump model weights (and biases).
"""

import argparse
import logging
import os
import re
import sys
import uuid

import numpy as np
from pydoop import hdfs
import tensorflow as tf

import pydeep.models as models
from pydeep.common import LOG_LEVELS

logging.basicConfig()
LOGGER = logging.getLogger("dump_weights")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", metavar="INPUT_DIR [INPUT_DIR...]")
    parser.add_argument("--architecture", metavar="STR",
                        default=models.DEFAULT)
    parser.add_argument("--collate", action="store_true")
    parser.add_argument("--log-level", metavar="|".join(LOG_LEVELS),
                        choices=LOG_LEVELS, default="INFO")
    parser.add_argument("--output", metavar="OUTPUT_DIR")
    return parser


def get_wb(model, path):
    """\
    Get weights and biases from the model checkpoint stored in path.
    """
    with tf.Session(graph=tf.Graph()) as session:
        models.load_checkpoint(path)
        graph = session.graph
        weights, biases = graph.get_collection("trainable_variables")
        if len(weights.shape) == 1:
            assert len(biases.shape) == 2
            weights, biases = biases, weights
        return session.run((weights, biases))


def get_all_wb(model, checkpoint_dir):
    """\
    Get all weights and biases from model checkpoints in checkpoint_dir.

    checkpoint_dir:
      part-m-00000.zip
      part-m-00001.zip
      ...

    return:
      {"00000": W0, "00001": W1, ...}, {"00000": b0, "00001": b1, ...}
    """
    paths = []
    tags = {}
    for p in hdfs.ls(checkpoint_dir):
        m = re.match(r"^part-m-(\d+)\.zip$", hdfs.path.basename(p))
        if m:
            paths.append(p)
            tags[p] = m.groups()[0]
    weights, biases = {}, {}
    for p in paths:
        t = tags[p]
        weights[t], biases[t] = get_wb(model, p)
        LOGGER.info("%s: W %r b %r", p, weights[t].shape, biases[t].shape)
    return weights, biases


def main(argv=sys.argv):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = make_parser()
    args = parser.parse_args(argv[1:])
    LOGGER.setLevel(args.log_level)
    if not args.output:
        args.output = "pydeep-%s" % uuid.uuid4()
    LOGGER.info("dumping to %s", args.output)
    hdfs.mkdir(args.output)
    model = models.get_model_info(args.architecture)
    if args.collate:
        all_w, all_b = {}, {}
    for d in args.input:
        bn = hdfs.path.basename(d)
        weights, biases = get_all_wb(model, d)
        if args.collate:
            all_w.update({"%s_%s" % (d, t): w for (t, w) in weights.items()})
            all_b.update({"%s_%s" % (d, t): b for (t, b) in biases.items()})
        else:
            w_path = hdfs.path.join(args.output, "%s_weights.npz" % bn)
            b_path = hdfs.path.join(args.output, "%s_biases.npz" % bn)
            with hdfs.open(w_path, "wb") as f:
                np.savez(f, **weights)
            with hdfs.open(b_path, "wb") as f:
                np.savez(f, **biases)
    if args.collate:
        with hdfs.open(hdfs.path.join(args.output, "weights.npz"), "wb") as f:
            np.savez(f, **all_w)
        with hdfs.open(hdfs.path.join(args.output, "biases.npz"), "wb") as f:
            np.savez(f, **all_b)


if __name__ == "__main__":
    sys.exit(main())
