"""\
Hadoop I/O classes.

The current implementation is somewhat minimalistic.
An avro implementation would be more elegant, but probably slower.
"""

import io
import logging
import struct

import numpy as np
import pydoop.mapreduce.api as api
import pydoop.hdfs as hdfs
from pydoop.utils.serialize import OpaqueInputSplit

from .common import GRAPH_ARCH_KEY
from .models import get_model_info


logging.basicConfig()
LOGGER = logging.getLogger("pydeep.ioformats")
LOGGER.setLevel(logging.INFO)


class SamplesReader(api.RecordReader):

    def __init__(self, context):
        super(SamplesReader, self).__init__(context)
        self.logger = LOGGER.getChild("SamplesReader")
        self.logger.debug('started')
        raw_split = context.get_input_split(raw=True)
        self.isplit = OpaqueInputSplit().read(io.BytesIO(raw_split))
        self.paths = self.isplit.payload
        self.n_paths = len(self.paths)

    def next(self):
        try:
            return 1, self.paths.pop()
        except IndexError:
            raise StopIteration

    def get_progress(self):
        return float(len(self.paths) / self.n_paths)


class BottleneckProjectionsWriter(api.RecordWriter):
    """\
    Write out each bottleneck as a raw binary dump.
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

    def emit(self, _, value):
        self.file.write(value.tobytes())


class BottleneckProjectionsReader(api.RecordReader):
    """Read binary records representing bottleneck projection of images.
    BROKEN
    """
    def __init__(self, context):
        super(BottleneckProjectionsReader, self).__init__(context)
        self.logger = LOGGER.getChild("BottleneckProjectionsReader")
        raw_split = context.get_input_split(raw=True)
        self.isplit = OpaqueInputSplit().read(io.BytesIO(raw_split))
        jc = context.job_conf
        model = get_model_info(jc[GRAPH_ARCH_KEY])
        self.bottleneck_tensor_size = model['bottleneck_tensor_size']
        self.record_length = 4 * self.bottleneck_tensor_size
        # now we need to manage creating batches from multiple files
        # remainder = self.isplit.offset % self.record_length
        # self.bytes_read = 0 if remainder == 0 else (self.record_length -
        #                                             remainder)
        # self.file = hdfs.open(self.isplit.filename)
        # self.file.seek(self.isplit.offset + self.bytes_read)

    def close(self):
        self.logger.debug("closing open handles")
        self.file.close()
        self.file.fs.close()

    def next(self):
        if self.bytes_read > self.isplit.length:
            raise StopIteration
        record = self.file.read(self.record_length)
        if not record:
            self.logger.debug("StopIteration on eof")
            raise StopIteration
        if len(record) < self.record_length:  # broken file?
            self.logger.warn("StopIteration on bad rec len %d", len(record))
            raise StopIteration
        self.bytes_read += self.record_length
        key = struct.unpack('>I', record[:4])
        value = np.fromstring(record[4:], dtype=np.float32,
                              count=self.bottleneck_tensor_size)
        return (key, value)

    def get_progress(self):
        return min(float(self.bytes_read) / self.isplit.length, 1.0)
