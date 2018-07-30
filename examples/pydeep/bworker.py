"""\
Calculate bottlenecks for all images in the input stream.

Input key: image path
Input value: image content
"""
from hashlib import md5

import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp

from pydeep.ioformats import WholeFileReader as Reader
from pydeep.ioformats import BottleneckProjectionsWriter as Writer
from pydeep.tflow import BottleneckProjector
from pydeep.common import GRAPH_ARCH_KEY
import pydeep.models as models


class Mapper(api.Mapper):

    def __init__(self, context):
        super(Mapper, self).__init__(context)
        jc = context.job_conf
        model = models.get_model_info(jc[GRAPH_ARCH_KEY])
        self.projector = BottleneckProjector(model)

    def close(self):
        self.projector.close_session()

    def map(self, context):
        # Here, if needed, we could also generate many derived
        # (distorted) variants of the image
        checksum = md5(context.value).digest()
        bneck = self.projector.project(context.value)
        cls = context.key.rsplit("/", 2)[1]
        context.emit(cls, (checksum, bneck))


factory = pp.Factory(mapper_class=Mapper, record_reader_class=Reader,
                     record_writer_class=Writer)


def __main__():
    pp.run_task(factory, auto_serialize=False)
