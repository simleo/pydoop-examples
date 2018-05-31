"""\
Information on known network architectures.
"""

import json
import pydoop.hdfs as hdfs


BASE_URL = "http://download.tensorflow.org/models"

INCEPTION_V3 = {
    'url': '%s/image/imagenet/inception-2015-12-05.tgz' % BASE_URL,
    'filename': 'classify_image_graph_def.pb',
    'path': 'inception_v3/classify_image_graph_def.pb',
    'prep_path': 'inception_v3/prep.pb',
    'retrained_path': 'inception_v3/retrained.pb',
    'bottleneck_tensor_size': 2048,
    'input_width': 299,
    'input_height': 299,
    'input_depth': 3,
    'input_mean': 128,
    'input_std': 128,
    'bottleneck_tensor_name': 'pool_3/_reshape:0',
    'resized_input_tensor_name': 'Mul:0',
    # filled by the initial graph setup step
    'jpg_input_tensor_name': None,
    'mul_image_tensor_name': None,
}

BY_NAME = {
    'inception_v3': INCEPTION_V3,
}

DEFAULT = 'inception_v3'


def get_model_info(arch):
    try:
        return BY_NAME[arch]
    except KeyError:
        raise ValueError("unknown architecture: %s" % arch)


def dump(model_info, path):
    with hdfs.open(path, 'wt') as f:
        json.dump(model_info, f)


def load(path):
    with hdfs.open(path, 'rt') as f:
        return json.load(f)


def get_info_path(path):
    return "%s.info" % path.rsplit(".", 1)[0]
