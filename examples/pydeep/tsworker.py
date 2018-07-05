"""\
Test each model on the whole dataset.

Input key: path to the serialized model
Input value: serialized model as a byte string
"""

import logging

import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp
import tensorflow as tf

from pydeep.ioformats import WholeFileReader, BottleneckStore
import pydeep.common as common

import pydeep.models as models

logging.basicConfig()
LOGGER = logging.getLogger("rsworker")


class PathNameReader(WholeFileReader):

    def path_to_kv(self, path):
        return None, path


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
        LOGGER.info("testing %s" % (context.value))
        with tf.Session(graph=tf.Graph()) as session:
            models.load_checkpoint(context.value)
            graph = session.graph
            eval_step, prediction, bneck_input, gtruth_input = (
                self.model.get_eval_step(graph),
                self.model.get_prediction(graph),
                self.model.get_bneck_input(graph),
                self.model.get_gtruth_input(graph),
            )
            test_accuracy, predictions = session.run(
                [eval_step, prediction],
                feed_dict={bneck_input: self.bnecks,
                           gtruth_input: self.gtruths})
        context.emit(context.value, str(test_accuracy))


factory = pp.Factory(mapper_class=Mapper, record_reader_class=PathNameReader)


def __main__():
    pp.run_task(factory)
