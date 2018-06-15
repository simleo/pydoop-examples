"""\
Download the graph def for the original model, add the JPEG decoding
subgraph and store its def.
"""

import argparse
import logging
import os
import sys

from pydoop import hdfs

import pydeep.tflow as tflow
import pydeep.models as models
from pydeep.common import LOG_LEVELS


logging.basicConfig()


def get_graph(model, log_level="INFO"):
    logger = logging.getLogger("graph_setup")
    logger.setLevel(log_level)
    if hdfs.path.exists(model["prep_path"]):
        logger.info("%s already exists, nothing to do" % model["prep_path"])
        return
    if hdfs.path.exists(model["path"]):
        logger.info("found original model graph at %s" % model["path"])
    else:
        logger.info("downloading original model graph")
        tflow.get_model_graph(model)
        models.dump(model, models.get_info_path(model["path"]))
    logger.info("adding JPEG decoding")
    graph = tflow.load_graph(model["path"])
    model = tflow.add_jpeg_decoding(model, graph)
    bneck_tensor = graph.get_tensor_by_name(model['bottleneck_tensor_name'])
    model['bottleneck_tensor_dtype'] = bneck_tensor.dtype.name
    models.dump(model, models.get_info_path(model["prep_path"]))
    logger.info("saving graph to %s" % model["prep_path"])
    tflow.save_graph(graph, model["prep_path"])


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", metavar="STR",
                        default=models.DEFAULT)
    parser.add_argument("--log-level", metavar="|".join(LOG_LEVELS),
                        choices=LOG_LEVELS, default="INFO")
    return parser


def main(argv=sys.argv):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = make_parser()
    args = parser.parse_args(argv[1:])
    model = models.get_model_info(args.architecture)
    return get_graph(model, log_level=args.log_level)


if __name__ == "__main__":
    sys.exit(main())
