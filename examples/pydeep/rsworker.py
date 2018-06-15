"""\
Retrain the model on the given bottleneck subsets.

Input key: training step index
Input value: input data for this step as a [(bottleneck, ground_truth)] list
"""

import os
import logging

import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp

from pydeep.ioformats import BottleneckProjectionsReader as Reader
# from pydeep.ioformats import TBDWriter as Writer
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
        model = models.get_model_info(jc[common.GRAPH_ARCH_KEY])
        model = models.load(models.get_info_path(model["pretrain_path"]))
        export_dir = os.path.abspath(jc[common.MODEL_EXPORT_DIR_KEY])
        self.retrainer = tflow.Retrainer(model, export_dir)

    def close(self):
        self.retrainer.close_session()

    def map(self, context):
        i = context.key
        LOGGER.debug('step #: %d', i)
        bottlenecks, ground_truths = zip(*context.value)
        self.retrainer.run_train_step(bottlenecks, ground_truths)
        if (i % self.eval_step_interval == 0) or (i + 1 >= self.n_steps):
            accuracy, cross_entropy = self.retrainer.run_eval_step(
                bottlenecks, ground_truths)
            LOGGER.info('step %d: accuracy = %.1f%%, cross entropy = %f' %
                        (i, 100 * accuracy, cross_entropy))
            context.emit(str(i).encode(), str(cross_entropy).encode())
        # TODO: add validation step
        # context.emit(TBD_k, TBD_v)


factory = pp.Factory(mapper_class=Mapper, record_reader_class=Reader)
# record_writer_class=Writer)


def __main__():
    pp.run_task(factory, auto_serialize=False)
