"""\
Compute predictions from Keras models.

The --models-tar arg expects a .tar archive of Keras model dumps (as JSON
model description + HDF5 weights)) at the top level, named as m1.{json,hdf5},
m2.{json,hdf5}, ...
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

import pydeep.arrayblob as arrayblob
import pydeep.common as common

LOGGER = logging.getLogger("keras_predictions")
WORKER = "kpworker"
PACKAGE = "pydeep"
MODELS_CACHE_LINK = "models"
CACHE_ARCHIVES_KEY = "mapreduce.job.cache.archives"
V1_CACHE_ARCHIVES_KEY = "mapred.cache.archives"


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps-per-file', metavar='INT', type=int, default=2)
    parser.add_argument('--models-tar', default="models.tar")
    add_parser_common_arguments(parser)
    add_parser_arguments(parser)
    return parser


def map_blobs(input_dir):
    m = {_["name"].rsplit("/", 1)[1]: _ for _ in hdfs.lsl(input_dir)}
    rval = {}
    for basename, stat in m.items():
        if stat["kind"] != "file" or basename.startswith("_"):
            continue
        base, ext = hdfs.path.splitext(basename)
        if ext == ".data":
            meta_bn = "%s.meta" % base
            if meta_bn not in m:
                raise RuntimeError("metafile for %r not found" % (basename,))
            with hdfs.open(m[meta_bn]["name"], "rt") as f:
                dtype, shape, recsize = arrayblob.read_meta(f)
            n_records, r = divmod(stat["size"], recsize)
            assert r == 0
            rval[base] = n_records
    return rval


def generate_input_splits(N, input_dir, splits_path):
    """\
    For each input file, assign a subset of arrays to each split.
    """
    m = map_blobs(input_dir)
    LOGGER.debug("%d files, %r records", len(m), sorted(m.values()))
    splits = []
    for base, L in m.items():
        if N > L:
            N = L
            LOGGER.warn("%r is too short, will only do %d splits", base, N)
        base = hdfs.path.join(input_dir, base)
        for offset, length in common.balanced_chunks(L, N):
            splits.append(OpaqueInputSplit(1, (base, offset, length)))
    LOGGER.debug("saving input splits to: %s", splits_path)
    with hdfs.open(splits_path, 'wb') as f:
        write_opaques(splits, f)
    return N


# workaround until we have better properties support in pydoop submit
def finalize_cache_archives(props, models_tar, models_link):
    archives = props.get(CACHE_ARCHIVES_KEY, "").strip(",")
    archives += props.pop(V1_CACHE_ARCHIVES_KEY, "").strip(",")
    if archives:
        archives += ","
    archives += "%s#%s" % (models_tar, models_link)
    props[CACHE_ARCHIVES_KEY] = archives


def main(argv=None):

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    wd = tempfile.mkdtemp(prefix="pydeep_")
    zip_fn = os.path.join(wd, "{}.zip".format(PACKAGE))
    shutil.make_archive(*zip_fn.rsplit(".", 1), base_dir=PACKAGE)

    parser = make_parser()
    args, unknown_args = parser.parse_known_args(argv)
    LOGGER.setLevel(args.log_level)
    if not hdfs.path.exists(args.models_tar):
        raise RuntimeError("%r not found on HDFS" % (args.models_tar))
    args.job_name = WORKER
    args.module = WORKER
    args.upload_file_to_cache = ['%s.py' % WORKER]
    args.python_zip = [zip_fn]
    args.do_not_use_java_record_reader = True
    args.do_not_use_java_record_writer = True
    args.num_reducers = 0

    splits_path = os.path.join(args.input, '_' + uuid.uuid4().hex)
    generate_input_splits(args.maps_per_file, args.input, splits_path)
    submitter = PydoopSubmitter()
    submitter.set_args(args, [] if unknown_args is None else unknown_args)
    finalize_cache_archives(
        submitter.properties, args.models_tar, MODELS_CACHE_LINK
    )
    submitter.properties.update({
        common.LOG_LEVEL_KEY: args.log_level,
        common.MODELS_DIR_KEY: MODELS_CACHE_LINK,
        common.PYDOOP_EXTERNALSPLITS_URI_KEY: splits_path,
    })

    submitter.run()

    hdfs.rmr(splits_path)
    shutil.rmtree(wd)


if __name__ == "__main__":
    main(sys.argv)
