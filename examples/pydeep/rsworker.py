"""\
Retrain the model on the given bottleneck subsets.

Input key: training step index
Input value: input data for this step as a [(bottleneck, ground_truth)] list
"""

import logging

import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp

from pydeep.ioformats import BottleneckProjectionsReader as Reader
import pydeep.tflow as tflow
import pydeep.common as common
import pydeep.models as models

logging.basicConfig()
LOGGER = logging.getLogger("rsworker")


class Mapper(api.Mapper):

    def __init__(self, context):
        super(Mapper, self).__init__(context)
        jc = context.job_conf
        LOGGER.setLevel(jc[common.LOG_LEVEL_KEY])
        self.n_steps = jc.get_int(common.NUM_STEPS_KEY)
        self.eval_step_interval = jc.get_int(common.EVAL_STEP_INTERVAL_KEY)
        learning_rate = jc.get_float(common.LEARNING_RATE_KEY)
        n_classes = jc.get_int(common.NUM_CLASSES_KEY)
        self.validation_percent = jc.get_int(common.VALIDATION_PERCENT_KEY)
        model = models.get_model_info(jc[common.GRAPH_ARCH_KEY])
        self.retrainer = tflow.Retrainer(model, n_classes, learning_rate)
        self.out_path = "%s.pb" % context.get_default_work_file()

    def close(self):
        self.retrainer.dump_output_graph(self.out_path)
        self.retrainer.close_session()

    def map(self, context):
        i = context.key
        split_i = round(len(context.value) * self.validation_percent / 100)
        val_set, train_set = context.value[:split_i], context.value[split_i:]
        LOGGER.debug('step %d: train size = %d, val size = %d',
                     i, len(train_set), len(val_set))
        train_bnecks, train_gtruths = zip(*train_set)
        val_bnecks, val_gtruths = zip(*val_set)
        self.retrainer.run_train_step(train_bnecks, train_gtruths)
        if (i % self.eval_step_interval == 0) or (i + 1 >= self.n_steps):
            train_accuracy, cross_entropy = self.retrainer.run_eval_step(
                train_bnecks, train_gtruths)
            LOGGER.info('step %d: train accuracy = %f%%, cross entropy = %f',
                        i, 100 * train_accuracy, cross_entropy)
            val_accuracy = self.retrainer.run_validation_step(
                val_bnecks, val_gtruths)
            LOGGER.info('step %d: validation accuracy = %f%%',
                        i, 100 * val_accuracy)
            context.emit(i, "%s\t%s\t%s" %
                         (cross_entropy, train_accuracy, val_accuracy))


factory = pp.Factory(mapper_class=Mapper, record_reader_class=Reader)


def __main__():
    pp.run_task(factory)
