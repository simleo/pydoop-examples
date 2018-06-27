"""\
Tensorflow model management.
"""

from collections import namedtuple
from mmap import PAGESIZE
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib

import pydoop.hdfs as hdfs
import tensorflow as tf


BASE_URL = "http://download.tensorflow.org/models"
INCEPTION_V3 = "inception_v3"
DEFAULT = INCEPTION_V3
MOBILENET_V1 = re.compile(r"^mobilenet_([0-9.]+)_(\d+)(.*)$")
MOBILENET_V1_VERSIONS = frozenset(("0.25", "0.50", "0.75", "1.0"))
MOBILENET_V1_INPUT_SIZES = frozenset(("128", "160", "192", "224"))

JPG_INPUT_NAME = "jpg_input_name"
MUL_IMAGE_NAME = "mul_image_name"


Input = namedtuple("Input", "width, height, depth, mean, std")
TensorNames = namedtuple("TensorNames", "bottleneck, resized_input")


class Model(namedtuple("Model", "name, url, filename, input, tensor_names")):

    __slots__ = ()

    @property
    def base_dir(self):
        return self.name

    @property
    def path(self):
        return hdfs.path.join(self.base_dir, self.filename)

    @property
    def prep_path(self):
        return hdfs.path.join(self.base_dir, "prep.meta")

    def get_bottleneck(self, graph):
        return graph.get_tensor_by_name(self.tensor_names.bottleneck)

    def get_resized_input(self, graph):
        return graph.get_tensor_by_name(self.tensor_names.resized_input)

    def load(self, path):
        graph_def = tf.GraphDef()
        with hdfs.open(path, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        return graph

    def load_meta(self, path):
        meta_graph_def = tf.MetaGraphDef()
        with hdfs.open(path, 'rb') as f:
            meta_graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.train.import_meta_graph(meta_graph_def)
        return graph

    def load_original(self):
        return self.load(self.path)

    def load_prep(self):
        return self.load_meta(self.prep_path)

    def save_meta(self, graph, path):
        meta_graph_def = tf.train.export_meta_graph(graph=graph)
        hdfs.dump(meta_graph_def.SerializeToString(), path)

    def save_prep(self, graph):
        self.save_meta(graph, self.prep_path)

    def download(self):
        tar_name = self.url.rsplit("/", 1)[-1]

        def _report(count, block_size, total_size):
            perc = 100 * count * block_size / total_size
            sys.stdout.write("\r>> Getting %s %.1f%%" % (tar_name, perc))
            sys.stdout.flush()

        tempd = tempfile.mkdtemp(prefix="pydeep_")
        tar_path = os.path.join(tempd, tar_name)
        tar_path, _ = urllib.request.urlretrieve(self.url, tar_path, _report)
        print()
        dest_dir = hdfs.path.dirname(self.path)
        if dest_dir:
            hdfs.mkdir(dest_dir)
        with tarfile.open(tar_path, "r:gz") as tar:
            try:
                info = tar.getmember(self.filename)
            except KeyError:
                raise ValueError("{} not found in {}".format(
                    self.filename, tar_name))
            f_in = tar.extractfile(info)
            with hdfs.open(self.path, "wb") as f_out:
                while True:
                    chunk = f_in.read(PAGESIZE)
                    if not chunk:
                        break
                    f_out.write(chunk)
        shutil.rmtree(tempd)

    def add_jpeg_decoding(self):
        graph = self.load_original()
        with tf.Session(graph=graph):
            jpg_input = tf.placeholder(tf.string, name=JPG_INPUT_NAME)
            dimage = tf.image.decode_jpeg(jpg_input, channels=self.input.depth)
            dimage_as_float = tf.cast(dimage, dtype=tf.float32)
            dimage4d = tf.expand_dims(dimage_as_float, 0)
            resize_shape = tf.stack([self.input.height, self.input.width])
            resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
            resized_image = tf.image.resize_bilinear(
                dimage4d, resize_shape_as_int)
            offset_image = tf.subtract(resized_image, self.input.mean)
            mul_image = tf.multiply(
                offset_image, 1.0 / self.input.std, name=MUL_IMAGE_NAME)
            graph.add_to_collection(JPG_INPUT_NAME, jpg_input)
            graph.add_to_collection(MUL_IMAGE_NAME, mul_image)
        self.save_prep(graph)


def get_model_info(name=INCEPTION_V3):
    name = name.lower()
    if name == INCEPTION_V3:
        return Model(
            name=name,
            url="%s/image/imagenet/inception-2015-12-05.tgz" % BASE_URL,
            filename="classify_image_graph_def.pb",
            input=Input(width=299, height=299, depth=3, mean=128, std=128),
            tensor_names=TensorNames(
                bottleneck="pool_3/_reshape:0", resized_input="Mul:0"
            )
        )
    else:
        match = MOBILENET_V1.match(name)
        if not match:
            raise ValueError("unknown architecture: %s" % name)
        version, input_size, quantized = match.groups()
        if version not in MOBILENET_V1_VERSIONS:
            raise ValueError("invalid mobilenet version: %s" % version)
        if input_size not in MOBILENET_V1_INPUT_SIZES:
            raise ValueError("invalid mobilenet input size: %s" % input_size)
        if quantized and not quantized == "_quantized":
            raise ValueError("unrecognized mobilenet option: %s" % quantized)
        url = "%s/mobilenet_v1_%s_%s_frozen.tgz" % (
            BASE_URL, version, input_size
        )
        filename = "%s_graph.pb" % ("quantized" if quantized else "frozen")
        w = h = int(input_size)
        input = Input(width=w, height=h, depth=3, mean=127.5, std=127.5)
        tensor_names = TensorNames(
            bottleneck="MobilenetV1/Predictions/Reshape:0",
            resized_input="input:0",
        )
        return Model(name, url, filename, input, tensor_names)
