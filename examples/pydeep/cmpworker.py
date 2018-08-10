"""\
Compare pairs of prediction sets.
"""

import logging

import numpy as np
import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp
import pydoop.hdfs as hdfs

from pydeep.ioformats import OpaqueListReader
import pydeep.arrayblob as arrayblob
import pydeep.common as common

logging.basicConfig()
LOGGER = logging.getLogger("cmpworker")


# TODO: generalize
def cmp_predictions(p1, p2):
    """\
    Compute a distance between two sets of predictions.

    Assumes p1, p2 are n_images x n_classes arrays, so row i =
    prediction (as a vector of probabilities for each class) for img i.
    """
    # Euclidean distance
    return np.linalg.norm(p1 - p2, axis=1)


class Writer(api.RecordWriter):

    def __init__(self, context):
        super(Writer, self).__init__(context)
        self.base_fn = context.get_default_work_file()
        self.tabf = hdfs.open(self.base_fn, "wt")
        self.writer = None

    def close(self):
        self.tabf.close()
        if self.writer is not None:
            self.writer.close()

    def emit(self, key, value):
        (b1, b2), d = key, value
        # tags are unique because they come from the same input dir
        t1 = b1.rsplit("/", 1)[1].rsplit(".", 1)[0]
        t2 = b2.rsplit("/", 1)[1].rsplit(".", 1)[0]
        self.tabf.write("%s\t%s\n" % (t1, t2))
        if self.writer is None:
            self.writer = arrayblob.Writer(self.base_fn, d.shape, d.dtype)
        self.writer.write(d)


class Mapper(api.Mapper):

    def __init__(self, context):
        super(Mapper, self).__init__(context)
        jc = context.job_conf
        LOGGER.setLevel(jc[common.LOG_LEVEL_KEY])

    def map(self, context):
        LOGGER.debug("%s vs %s" % tuple(context.value))
        pred_sets = []
        for blob in context.value:
            with arrayblob.Reader(blob) as reader:
                pred_sets.append(np.stack(reader))
        context.emit(context.value, cmp_predictions(*pred_sets))


factory = pp.Factory(
    mapper_class=Mapper,
    record_reader_class=OpaqueListReader,
    record_writer_class=Writer,
)


def __main__():
    pp.run_task(factory)
