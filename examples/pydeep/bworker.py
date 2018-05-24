"""\
Calculate bottlenecks for all images in the input stream.

Input key: image label
Input value: image path
"""

import pydoop.mapreduce.api as api
import pydoop.mapreduce.pipes as pp

from pydeep.ioformats import SamplesReader as Reader
from pydeep.ioformats import BottleneckProjectionsWriter as Writer
from pydeep.tflow import BottleneckProjector
from pydeep.models import model
from pydeep.keys import GRAPH_ARCH_KEY, GRAPH_PATH_KEY


class Mapper(api.Mapper):

    def __init__(self, context):
        super(Mapper, self).__init__(context)
        jc = context.job_conf
        m = model[jc[GRAPH_ARCH_KEY]]
        m['path'] = jc[GRAPH_PATH_KEY]
        self.projector = BottleneckProjector(m)

    def map(self, context):
        # Here, if needed, we could also generate many derived
        # (distorted) variants of the image
        context.emit(context.key, self.projector(context.value))


factory = pp.Factory(mapper_class=Mapper, record_reader_class=Reader,
                     record_writer_class=Writer)


def __main__():
    pp.run_task(factory, auto_serialize=False)
