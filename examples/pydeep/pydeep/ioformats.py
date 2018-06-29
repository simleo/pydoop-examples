"""\
I/O classes.

Note on naming: use 'length' (or 'len') for number of elements and 'size' for
size in bytes.
"""

from hashlib import md5
import io
import logging
import random
from collections import defaultdict

import numpy as np
import pydoop.mapreduce.api as api
import pydoop.hdfs as hdfs
from pydoop.utils.serialize import OpaqueInputSplit
import tensorflow as tf

import pydeep.common as common
import pydeep.models as models


logging.basicConfig()
LOGGER = logging.getLogger("pydeep.ioformats")
LOGGER.setLevel(logging.INFO)

CHECKSUM_SIZE = md5().digest_size


class FileCache(defaultdict):

    def __missing__(self, path):
        f = hdfs.open(path, "rb")
        self[path] = f
        return f


class BottleneckStore(object):

    def __init__(self, bneck_len, bneck_dtype):
        bneck_dtype = tf.as_dtype(bneck_dtype)
        self.bneck_len = bneck_len
        self.bneck_size = bneck_dtype.size * self.bneck_len
        self.record_size = CHECKSUM_SIZE + self.bneck_size
        self.dtype = bneck_dtype.as_numpy_dtype

    def get_bnecks(self, top_dir, posmap=None, checksums=False):
        top_dir = top_dir.rstrip("/")
        if posmap is None:
            posmap = self.build_map(top_dir)
        fcache = FileCache()
        ret = {}
        bneck_size, dtype = self.bneck_size, self.dtype
        for cls, positions in posmap.items():
            bnecks = ret[cls] = []
            for name, offset in positions:
                path = "%s/%s/%s" % (top_dir, cls, name)
                chunk = fcache[path].pread(offset + CHECKSUM_SIZE, bneck_size)
                bneck = np.frombuffer(chunk, dtype)
                if checksums:
                    csum = fcache[path].pread(offset, CHECKSUM_SIZE)
                    bnecks.append((csum, bneck))
                else:
                    bnecks.append(bneck)
        for f in fcache.values():
            f.close()
        return ret

    def build_map(self, top_dir):
        """\
        For each subdir (corresponding to an image class), build the full
        list of (filename, offset) pair where each bottleneck dump can be
        retrieved.

        {'dandelion': [
            ('part-m-00000', 0),
            ('part-m-00000', 8192),
            ...
            ('part-m-00003', 163840)
        ],
        'roses': [
            ('part-m-00000', 0),
            ...
        ]}
        """
        m = {}
        basename = hdfs.path.basename
        with hdfs.hdfs() as fs:
            for stat in fs.list_directory(top_dir):
                if stat['kind'] != 'directory':
                    continue
                subd = stat['name']
                positions = []
                for s in fs.list_directory(subd):
                    bname = basename(s["name"])
                    if bname.startswith("_"):
                        continue
                    assert s["size"] % self.record_size == 0
                    for i in range(0, s["size"], self.record_size):
                        positions.append((bname, i))
                m[basename(subd)] = positions
        return m

    @staticmethod
    def assign_labels(top_dir):
        classes = [hdfs.path.basename(_["name"]) for _ in hdfs.lsl(top_dir)
                   if _["kind"] == "directory"]
        return {c: i for i, c in enumerate(sorted(classes))}

    @staticmethod
    def bnecks_map_to_vectors(bnecks_map, labels):
        all_bnecks, all_ground_truths = [], []
        for c, bnecks in bnecks_map.items():
            all_bnecks.extend(bnecks)
            for i in range(len(bnecks)):
                gt = np.zeros(len(labels), dtype=np.float32)
                gt[labels[c]] = 1
                all_ground_truths.append(gt)
        return all_bnecks, all_ground_truths


class WholeFileReader(api.RecordReader):
    """\
    Assumes input split = list of HDFS file paths. Reads the *whole* content C
    of each file (the user is responsible for ensuring this fits in memory)
    and emits a (path, C) record.
    """
    def __init__(self, context):
        super(WholeFileReader, self).__init__(context)
        self.logger = LOGGER.getChild("WholeFileReader")
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
        return self.path_to_kv(path)

    def path_to_kv(self, path):
        with self.fs.open_file(path, 'rb') as f:
            data = f.read()
        return path, data

    def get_progress(self):
        return float(len(self.paths) / self.n_paths)


class SamplesReader(WholeFileReader):
    """\
    For each HDFS path in the input split, read its content C and emit an
    (md5(C), C) record.
    """
    def path_to_kv(self, path):
        _, data = super(SamplesReader, self).path_to_kv(path)
        return md5(data).digest(), data


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
        split = OpaqueInputSplit().read(io.BytesIO(raw_split))
        jc = context.job_conf
        model = models.get_model_info(jc[common.GRAPH_ARCH_KEY])
        graph = model.load_prep()
        bneck_tensor = model.get_bottleneck(graph)
        self.bneck_store = BottleneckStore(
            bneck_tensor.shape[1].value, bneck_tensor.dtype
        )
        self.n_steps = jc.get_int(common.NUM_STEPS_KEY)
        top_dir = jc.get(common.BNECKS_DIR_KEY)
        val_fraction = jc.get_int(common.VALIDATION_PERCENT_KEY) / 100
        # get *all* bottlenecks for this split, assuming they fit in memory
        bneck_map = self.bneck_store.get_bnecks(top_dir, posmap=split.payload)
        self.val_bneck_map, self.train_bneck_map = {}, {}
        while bneck_map:
            c, bnecks = bneck_map.popitem()
            i = round(val_fraction * len(bnecks))
            self.val_bneck_map[c] = bnecks[:i]
            self.train_bneck_map[c] = bnecks[i:]
        self.logger.info(
            "training size = %d, validation size = %d",
            len(self.train_bneck_map[c]), len(self.val_bneck_map[c])
        )
        train_bs = jc.get_int(common.TRAIN_BATCH_SIZE_KEY)
        val_bs = jc.get_int(common.VALIDATION_BATCH_SIZE_KEY)
        self.val_bs_map, self.train_bs_map = self.__map_bs(val_bs, train_bs)
        self.step_count = 0

    def __map_bs(self, val_bs, train_bs):
        val_bs_map, train_bs_map = {}, {}
        for c, bnecks in self.val_bneck_map.items():
            capped = min(val_bs, len(bnecks))
            val_bs_map[c] = capped if capped > 0 else len(bnecks)
        for c, bnecks in self.train_bneck_map.items():
            train_bs_map[c] = min(train_bs, len(bnecks))
        return val_bs_map, train_bs_map

    def next(self):
        if self.step_count >= self.n_steps:
            raise StopIteration
        train_batch = {c: random.sample(bnecks, self.train_bs_map[c])
                       for c, bnecks in self.train_bneck_map.items()}
        val_batch = {c: random.sample(bnecks, self.val_bs_map[c])
                     for c, bnecks in self.val_bneck_map.items()}
        self.step_count += 1
        return self.step_count, (train_batch, val_batch)

    def get_progress(self):
        return min(self.step_count / self.n_steps, 1.0)
