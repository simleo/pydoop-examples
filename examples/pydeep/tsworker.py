"""\
Test each model on the whole dataset.

Input key: path to the serialized model
Input value: serialized model as a byte string
"""

import logging

import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp
import tensorflow as tf

from pydeep.ioformats import WholeFileReader as Reader, BottleneckStore
import pydeep.common as common

import pydeep.models as models

logging.basicConfig()
LOGGER = logging.getLogger("rsworker")


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
            bneck_tensor.shape[1].value, bneck_tensor.dtype
        )
        # get *all* bottlenecks
        bneck_map = self.bneck_store.get_bnecks(top_dir)
        self.bnecks, self.gtruths = BottleneckStore.bnecks_map_to_vectors(
            bneck_map, BottleneckStore.assign_labels(top_dir)
        )

    def map(self, context):
        meta_graph_def = tf.MetaGraphDef()
        meta_graph_def.ParseFromString(context.value)
        with tf.Graph().as_default() as graph:
            tf.train.import_meta_graph(meta_graph_def)
        eval_step, prediction, bneck_input, gtruth_input = (
            self.model.get_eval_step(graph),
            self.model.get_prediction(graph),
            self.model.get_bneck_input(graph),
            self.model.get_gtruth_input(graph),
        )
        with tf.Session(graph=graph) as session:
            test_accuracy, predictions = session.run(
                [eval_step, prediction],
                feed_dict={bneck_input: self.bnecks,
                           gtruth_input: self.gtruths})
        context.emit(context.key, str(test_accuracy))


factory = pp.Factory(mapper_class=Mapper, record_reader_class=Reader)


def __main__():
    pp.run_task(factory)
