"""\
Hadoop I/O classes.

The current implementation is somewhat minimalistic.
An avro implementation would be more elegant, but probably slower.
"""

from hashlib import md5
import io
import logging
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf
import pydoop.mapreduce.api as api
import pydoop.hdfs as hdfs
from pydoop.utils.serialize import OpaqueInputSplit

from .common import GRAPH_ARCH_KEY, NUM_STEPS_KEY, TRAIN_BATCH_SIZE_KEY
from .models import get_model_info, load as load_model, get_info_path


logging.basicConfig()
LOGGER = logging.getLogger("pydeep.ioformats")
LOGGER.setLevel(logging.INFO)

CHECKSUM_LEN = md5().digest_size


class FileCache(defaultdict):

    def __missing__(self, path):
        f = hdfs.open(path, "rb")
        self[path] = f
        return f


class SamplesReader(api.RecordReader):
    """\
    For each HDFS path in the input split, read its content C and emit an
    (md5(C), C) record.
    """
    def __init__(self, context):
        super(SamplesReader, self).__init__(context)
        self.logger = LOGGER.getChild("SamplesReader")
        self.logger.debug('started')
        raw_split = context.get_input_split(raw=True)
        self.isplit = OpaqueInputSplit().read(io.BytesIO(raw_split))
        self.paths = self.isplit.payload
        self.n_paths = len(self.paths)
        self.fs = hdfs.hdfs()

    def close(self):
        self.fs.close()

    def next(self):
        try:
            path = self.paths.pop()
        except IndexError:
            raise StopIteration
        with self.fs.open_file(path, 'rb') as f:
            data = f.read()
        return md5(data).digest(), data

    def get_progress(self):
        return float(len(self.paths) / self.n_paths)


class BottleneckProjectionsWriter(api.RecordWriter):
    """\
    Write out a binary record for each bottleneck. Expects a bytes object (md5
    digest of the JPEG data) as the key and a numpy array (the bottleneck) as
    the value.
    """
    def __init__(self, context):
        super(BottleneckProjectionsWriter, self).__init__(context)
        self.logger = LOGGER.getChild("BottleneckProjectionsWriter")
        out_path = context.get_default_work_file()
        hdfs_user = context.job_conf.get("pydoop.hdfs.user", None)
        self.file = hdfs.open(out_path, "wb", user=hdfs_user)

    def close(self):
        self.logger.debug("closing open handles")
        self.file.close()
        self.file.fs.close()

    def emit(self, key, value):
        self.file.write(key + value.tobytes())


class BottleneckProjectionsReader(api.RecordReader):

    def __init__(self, context):
        super(BottleneckProjectionsReader, self).__init__(context)
        self.logger = LOGGER.getChild("BottleneckProjectionsReader")
        raw_split = context.get_input_split(raw=True)
        self.isplit = OpaqueInputSplit().read(io.BytesIO(raw_split))
        self.bneck_maps = self.isplit.payload
        jc = context.job_conf
        model = get_model_info(jc[GRAPH_ARCH_KEY])
        model = load_model(get_info_path(model["pretrain_path"]))
        self.dtype = tf.as_dtype(model['bottleneck_tensor_dtype'])
        self.length = self.dtype.size * model['bottleneck_tensor_size']
        self.n_steps = jc.get_int(NUM_STEPS_KEY)
        self.batch_size = jc.get_int(TRAIN_BATCH_SIZE_KEY)
        self.step_count = 0
        self.fcache = FileCache()
        self.labels = {d: i for i, d in enumerate(sorted(self.bneck_maps))}

    def close(self):
        self.logger.debug("closing open handles")
        for f in self.fcache.values():
            f.close()
        next(self.fcache.values()).fs.close()

    def next(self):
        if self.step_count >= self.n_steps:
            raise StopIteration
        record = []
        batch_size, fcache, length = self.batch_size, self.fcache, self.length
        dtype = self.dtype.as_numpy_dtype
        n_classes = len(self.bneck_maps)
        for subd, bneck_positions in self.bneck_maps.items():
            sample = random.sample(
                bneck_positions, min(batch_size, len(bneck_positions))
            )
            for name, offset in sample:
                path = "%s/%s" % (subd, name)
                chunk = fcache[path].pread(offset + CHECKSUM_LEN, length)
                bneck = np.frombuffer(chunk, dtype)
                gtruth = np.zeros(n_classes, dtype=np.float32)
                gtruth[self.labels[subd]] = 1
                record.append((bneck, gtruth))
        self.step_count += 1
        return self.step_count, record

    def get_progress(self):
        return min(self.step_count / self.n_steps, 1.0)
