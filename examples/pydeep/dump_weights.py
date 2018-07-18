"""\
Dump model weights (and biases).
"""

import argparse
import logging
import os
import re
import shutil
import sys
import tempfile
import uuid

import numpy as np
import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pipes
from pydoop import hdfs
from pydoop.app.submit import PydoopSubmitter
from pydoop.utils.serialize import OpaqueInputSplit, write_opaques
import tensorflow as tf

import pydeep.models as models
import pydeep.common as common
import pydeep.ioformats as ioformats

logging.basicConfig()
LOGGER = logging.getLogger("dump_weights")
DEFAULT_NUM_MAPS = 4
PACKAGE = "pydeep"


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", metavar="INPUT_DIR [INPUT_DIR...]")
    parser.add_argument("--input-from-list", action="store_true")
    parser.add_argument("--architecture", metavar="STR",
                        default=models.DEFAULT)
    parser.add_argument("--collate", action="store_true")
    parser.add_argument("--mapred", action="store_true")
    parser.add_argument('--num-maps', metavar='INT', type=int,
                        default=DEFAULT_NUM_MAPS)
    parser.add_argument("--log-level", metavar="|".join(common.LOG_LEVELS),
                        choices=common.LOG_LEVELS, default="INFO")
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


def run_locally(model, input_dirs, output_dir, collate=False):
    hdfs.mkdir(output_dir)
    if collate:
        all_w, all_b = {}, {}
    for d in input_dirs:
        bn = hdfs.path.basename(d)
        weights, biases = get_all_wb(model, d)
        if collate:
            all_w.update({"%s_%s" % (d, t): w for (t, w) in weights.items()})
            all_b.update({"%s_%s" % (d, t): b for (t, b) in biases.items()})
        else:
            w_path = hdfs.path.join(output_dir, "%s_weights.npz" % bn)
            b_path = hdfs.path.join(output_dir, "%s_biases.npz" % bn)
            with hdfs.open(w_path, "wb") as f:
                np.savez(f, **weights)
            with hdfs.open(b_path, "wb") as f:
                np.savez(f, **biases)
    if collate:
        with hdfs.open(hdfs.path.join(output_dir, "weights.npz"), "wb") as f:
            np.savez(f, **all_w)
        with hdfs.open(hdfs.path.join(output_dir, "biases.npz"), "wb") as f:
            np.savez(f, **all_b)


def run_mapred(model, input_dirs, output_dir, nmaps, log_level, collate=False):
    wd = tempfile.mkdtemp(prefix="pydeep_")
    zip_fn = os.path.join(wd, "{}.zip".format(PACKAGE))
    shutil.make_archive(*zip_fn.rsplit(".", 1), base_dir=PACKAGE)
    splits = common.balanced_split(input_dirs, nmaps)
    splits_uri = "pydoop_splits_%s" % uuid.uuid4().hex
    with hdfs.open(splits_uri, 'wb') as f:
        write_opaques([OpaqueInputSplit(1, _) for _ in splits], f)
    submitter = PydoopSubmitter()
    properties = {
        common.GRAPH_ARCH_KEY: model.name,
        common.LOG_LEVEL_KEY: log_level,
        common.NUM_MAPS_KEY: nmaps,
        common.PYDOOP_EXTERNALSPLITS_URI_KEY: splits_uri,
    }
    submitter.set_args(argparse.Namespace(
        D=list(properties.items()),
        avro_input=None,
        avro_output=None,
        cache_archive=None,
        cache_file=None,
        disable_property_name_conversion=True,
        do_not_use_java_record_reader=True,
        do_not_use_java_record_writer=True,
        entry_point="__main__",
        hadoop_conf=None,
        input=input_dirs[0],  # does it matter?
        input_format=None,
        job_conf=None,
        job_name="dump_weights",
        keep_wd=False,
        libjars=None,
        log_level=log_level,
        module=os.path.splitext(os.path.basename(__file__))[0],
        no_override_env=False,
        no_override_home=False,
        no_override_ld_path=False,
        no_override_path=False,
        no_override_pypath=False,
        num_reducers=0,
        output=output_dir,
        output_format=None,
        pretend=False,
        pstats_dir=None,
        python_program=sys.executable,
        python_zip=[zip_fn],
        set_env=None,
        upload_archive_to_cache=None,
        upload_file_to_cache=[__file__],
    ))
    submitter.run()
    hdfs.rmr(splits_uri)
    if collate:
        collate_mapred_output(output_dir)
    shutil.rmtree(wd)


def collate_mapred_output(output_dir):
    data = {"weights": {}, "biases": {}}
    pattern = re.compile(r"part-m-\d+-(\d+)-(weights|biases).npz")
    for path in hdfs.ls(output_dir):
        LOGGER.debug("processing: %s", path)
        m = pattern.match(hdfs.path.basename(path))
        if not m:
            continue
        seed, what = m.groups()
        with hdfs.open(path, "rb") as f:
            npzf = np.load(f)
            data[what].update(
                {"%s_%s" % (seed, t): w for (t, w) in npzf.iteritems()}
            )
    for k, v in data.items():
        out_path = hdfs.path.join(output_dir, "%s.npz" % k)
        LOGGER.info("saving collated %s to %s", k, out_path)
        with hdfs.open(out_path, "wb") as f:
            np.savez(f, **v)


class Reader(ioformats.WholeFileReader):

    def path_to_kv(self, path):
        return None, path


class Mapper(api.Mapper):

    def __init__(self, context):
        super(Mapper, self).__init__(context)
        jc = context.job_conf
        self.model = models.get_model_info(jc[common.GRAPH_ARCH_KEY])
        LOGGER.setLevel(jc[common.LOG_LEVEL_KEY])

    def map(self, context):
        checkpoint_dir = context.value
        bn = hdfs.path.basename(checkpoint_dir)
        weights, biases = get_all_wb(self.model, checkpoint_dir)
        context.emit(bn, (weights, biases))


class Writer(api.RecordWriter):

    def __init__(self, context):
        super(Writer, self).__init__(context)
        LOGGER.setLevel(context.job_conf[common.LOG_LEVEL_KEY])
        self.user = context.job_conf.get("pydoop.hdfs.user", None)
        self.base_path = context.get_default_work_file()
        LOGGER.info("base path: %r", self.base_path)

    def emit(self, key, value):
        weights, biases = value
        weights_path = "%s-%s-weights.npz" % (self.base_path, key)
        biases_path = "%s-%s-biases.npz" % (self.base_path, key)
        LOGGER.debug("writing weights to %s", weights_path)
        with hdfs.open(weights_path, "wb", user=self.user) as f:
            np.savez(f, **weights)
        LOGGER.debug("writing biases to %s", biases_path)
        with hdfs.open(biases_path, "wb", user=self.user) as f:
            np.savez(f, **biases)


def __main__():
    pipes.run_task(pipes.Factory(
        Mapper, record_writer_class=Writer, record_reader_class=Reader
    ))


def main(argv=sys.argv):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = make_parser()
    args = parser.parse_args(argv[1:])
    LOGGER.setLevel(args.log_level)
    if args.input_from_list:
        if len(args.input) > 1:
            raise RuntimeError(
                "with --input-from-list, specify only 1 input (file)"
            )
        with hdfs.open(args.input[0], "rt") as f:
            args.input = [_.strip() for _ in f]
    if not args.output:
        args.output = "pydeep-%s" % uuid.uuid4()
    LOGGER.info("dumping to %s", args.output)
    model = models.get_model_info(args.architecture)
    if args.mapred:
        run_mapred(model, args.input, args.output, args.num_maps,
                   args.log_level, collate=args.collate)
    else:
        run_locally(model, args.input, args.output, collate=args.collate)


if __name__ == "__main__":
    sys.exit(main())
