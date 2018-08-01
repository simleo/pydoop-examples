"""\
Test each model on the whole dataset.
"""

import logging

import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp
import pydoop.hdfs as hdfs
import tensorflow as tf

from pydeep.ioformats import WholeFileReader, BottleneckStore
import pydeep.arrayblob as arrayblob
import pydeep.common as common
import pydeep.models as models

logging.basicConfig()
LOGGER = logging.getLogger("rsworker")


class PathNameReader(WholeFileReader):

    def path_to_kv(self, path):
        return None, path


class TestResultsWriter(api.RecordWriter):

    def __init__(self, context):
        super(TestResultsWriter, self).__init__(context)
        tab_fn = context.get_default_work_file()
        self.d = tab_fn.rsplit("/", 1)[0]
        self.tabf = hdfs.open(tab_fn, "wt")

    def close(self):
        self.tabf.close()

    def emit(self, key, value):
        path, (test_accuracy, float_predictions) = key, value
        # tags are unique because they come from the same input dir
        tag = path.rsplit("/", 1)[1].rsplit(".", 1)[0]
        self.tabf.write("%s\t%s\n" % (tag, str(test_accuracy)))
        data_fn = hdfs.path.join(self.d, "%s.data" % tag)
        meta_fn = hdfs.path.join(self.d, "%s.meta" % tag)
        shape, dtype = float_predictions[0].shape, float_predictions[0].dtype
        with hdfs.open(data_fn, "wb") as df, hdfs.open(meta_fn, "wt") as mf:
            writer = arrayblob.Writer(df, mf, shape, dtype)
            for p in float_predictions:
                writer.write(p)


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
        LOGGER.info("testing %s" % (context.value))
        with tf.Session(graph=tf.Graph()) as session:
            models.load_checkpoint(context.value)
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
        context.emit(context.value, (test_accuracy, float_predictions))


factory = pp.Factory(
    mapper_class=Mapper,
    record_reader_class=PathNameReader,
    record_writer_class=TestResultsWriter,
)


def __main__():
    pp.run_task(factory)
