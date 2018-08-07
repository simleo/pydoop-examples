"""\
Compare pairs of models.
"""

import logging

import numpy as np
import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp
import pydoop.hdfs as hdfs
import tensorflow as tf

from pydeep.ioformats import OpaqueListReader, BottleneckStore
import pydeep.arrayblob as arrayblob
import pydeep.common as common
import pydeep.models as models

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
        (m1_path, m2_path), (acc1, acc2, d) = key, value
        # tags are unique because they come from the same input dir
        t1 = m1_path.rsplit("/", 1)[1].rsplit(".", 1)[0]
        t2 = m2_path.rsplit("/", 1)[1].rsplit(".", 1)[0]
        self.tabf.write("%s\t%s\t%s\t%s\n" % (t1, str(acc1), t2, str(acc2)))
        if self.writer is None:
            self.writer = arrayblob.Writer(self.base_fn, d.shape, d.dtype)
        self.writer.write(d)


class Mapper(api.Mapper):

    def __init__(self, context):
        super(Mapper, self).__init__(context)
        jc = context.job_conf
        LOGGER.setLevel(jc[common.LOG_LEVEL_KEY])
        top_dir = jc.get(common.BNECKS_DIR_KEY)
        self.model = models.get_model_info(jc[common.GRAPH_ARCH_KEY])
        graph = self.model.load_prep()
        bneck_tensor = self.model.get_bottleneck(graph)
        del graph
        self.bneck_store = BottleneckStore(
            top_dir, bneck_tensor.shape[1].value, bneck_tensor.dtype
        )
        bneck_map = self.bneck_store.get_all_bnecks()
        self.bnecks, self.gtruths = BottleneckStore.bnecks_map_to_vectors(
            bneck_map, BottleneckStore.assign_labels(top_dir)
        )
        # FIXME: use the checksums?
        self.bnecks = [_[1] for _ in self.bnecks]

    def map(self, context):
        m1, m2 = context.value
        LOGGER.info("%s vs %s" % (m1, m2))
        acc1, p1 = self.__run_model(m1)
        acc2, p2 = self.__run_model(m2)
        context.emit(context.value, (acc1, acc2, cmp_predictions(p1, p2)))

    def __run_model(self, model_path):
        with tf.Session(graph=tf.Graph()) as session:
            models.load_checkpoint(model_path)
            graph = session.graph
            eval_step, final_tensor, bneck_input, gtruth_input = (
                self.model.get_eval_step(graph),
                self.model.get_final_tensor(graph),
                self.model.get_bneck_input(graph),
                self.model.get_gtruth_input(graph),
            )
            test_accuracy, float_predictions = session.run(
                [eval_step, final_tensor],
                feed_dict={bneck_input: self.bnecks,
                           gtruth_input: self.gtruths})
        # float_predictions.shape = (n_images, n_classes)
        return test_accuracy, float_predictions


factory = pp.Factory(
    mapper_class=Mapper,
    record_reader_class=OpaqueListReader,
    record_writer_class=Writer,
)


def __main__():
    pp.run_task(factory)
