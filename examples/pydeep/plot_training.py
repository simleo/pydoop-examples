"""\
Plot stats output by retrain_subsets.py.

All paths are assumed to be HDFS paths unless they start with "file:/"
"""
import argparse
import logging
import os
import re
import sys

from pydoop import hdfs

from pydeep.common import LOG_LEVELS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOGGER = logging.getLogger("plot_training")

FORMAT = "png"
DPI = 300


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_path", metavar="IN_PATH")
    parser.add_argument("out_dir", metavar="OUT_DIR")
    parser.add_argument("--log-level", metavar="|".join(LOG_LEVELS),
                        choices=LOG_LEVELS, default="INFO")
    return parser


def gen_plots(paths, out_dir):
    hdfs.mkdir(out_dir)
    for p in paths:
        out_p = "%s.%s" % (
            hdfs.path.join(out_dir, hdfs.path.basename(p)), FORMAT
        )
        plot_stats(p, out_p)


def plot_stats(path, out_path):
    LOGGER.info("processing: %s", path)
    with hdfs.open(path, "rt") as f:
        data = [[float(_) for _ in line.strip().split("\t")] for line in f]
    step, cross_entropy, train_acc, val_acc = zip(*data)
    plt.cla()
    plt.clf()
    plt.plot(step, cross_entropy, label="cross entropy")
    plt.plot(step, train_acc, label="training accuracy")
    plt.plot(step, val_acc, label="validation accuracy")
    plt.xlabel("step")
    plt.legend()
    with hdfs.open(out_path, "wb") as f:
        plt.savefig(f, dpi=DPI, format=FORMAT)


def main(argv=sys.argv):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = make_parser()
    args = parser.parse_args(argv[1:])
    logging.basicConfig()
    LOGGER.setLevel(args.log_level)
    paths = None
    if hdfs.path.isfile(args.stats_path):
        paths = [args.stats_path]
    else:
        try:
            ls = hdfs.ls(args.stats_path)
        except IOError as e:
            return "ERROR: %s: %s" % (args.stats_path, e)
        paths = [_ for _ in ls if
                 re.match(r"^part-m-\d+$", hdfs.path.basename(_))]
    gen_plots(paths, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
