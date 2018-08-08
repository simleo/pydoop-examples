"""\
Compute predictions from Keras models.
"""

from collections import defaultdict
import logging
import os

from keras.models import model_from_json
import numpy as np
import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp
import pydoop.hdfs as hdfs

import pydeep.arrayblob as arrayblob
from pydeep.ioformats import ArrayBlobReader
import pydeep.common as common

logging.basicConfig()
LOGGER = logging.getLogger("kpworker")


class WriterCache(defaultdict):

    def __missing__(self, args):
        writer = arrayblob.Writer(*args)
        self[args] = writer
        return writer


class Writer(api.RecordWriter):

    def __init__(self, context):
        super(Writer, self).__init__(context)
        self.d, self.bn = context.get_default_work_file().rsplit("/", 1)
        self.cache = WriterCache()

    def close(self):
        for writer in self.cache.values():
            writer.close()

    def emit(self, key, value):
        path = hdfs.path.join(self.d, "%s-%s" % (key, self.bn))
        for pred in value:
            self.cache[(path, pred.shape, pred.dtype)].write(pred)


def load_models(models_dir):
    rval = {}
    contents = os.listdir(models_dir)
    LOGGER.debug("models dir list: %r", contents)
    for name in contents:
        head, tail = os.path.splitext(name)
        if tail != ".json":
            continue
        model_path = os.path.join(models_dir, name)
        weights_path = os.path.join(models_dir, "%s.hdf5" % head)
        if not os.path.isfile(weights_path):
            LOGGER.error("%r: weights not found, skipping", head)
            continue
        with open(model_path, "rt") as f:
            model = model_from_json(f.read())
        model.load_weights(weights_path)
        rval[head] = model
    return rval


class Mapper(api.Mapper):

    def __init__(self, context):
        super(Mapper, self).__init__(context)
        jc = context.job_conf
        LOGGER.setLevel(jc[common.LOG_LEVEL_KEY])
        self.models = load_models(jc[common.MODELS_DIR_KEY])
        LOGGER.info("models: %r", sorted(self.models))

    def map(self, context):
        for tag, model in self.models.items():
            # TODO: allow batches of arbitrary size
            a = context.value[np.newaxis, ...]
            pred = model.predict(a)
            context.emit(tag, pred)


factory = pp.Factory(
    mapper_class=Mapper,
    record_reader_class=ArrayBlobReader,
    record_writer_class=Writer,
)


def __main__():
    pp.run_task(factory)
