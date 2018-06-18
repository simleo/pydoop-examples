"""\
Create, store, load and execute graphs needed to retrain the network.

TODO: use the tf.data API
"""

from mmap import PAGESIZE
import os
import shutil
import sys
import tarfile
import tempfile
import urllib

import numpy as np
import pydoop.hdfs as hdfs
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import tag_constants, signature_constants


def get_model_graph(model):
    tar_name = model['url'].rsplit('/', 1)[-1]

    def _report(count, block_size, total_size):
        perc = 100 * count * block_size / total_size
        sys.stdout.write('\r>> Getting %s %.1f%%' % (tar_name, perc))
        sys.stdout.flush()

    tempd = tempfile.mkdtemp(prefix="pydeep_")
    tar_path = os.path.join(tempd, tar_name)
    tar_path, _ = urllib.request.urlretrieve(model['url'], tar_path, _report)
    print()
    dest_dir = hdfs.path.dirname(model['path'])
    if dest_dir:
        hdfs.mkdir(dest_dir)
    with tarfile.open(tar_path, 'r:gz') as tar:
        try:
            info = tar.getmember(model['filename'])
        except KeyError:
            raise ValueError("{} not found in {}".format(
                model['filename'], tar_name))
        f_in = tar.extractfile(info)
        with hdfs.open(model['path'], 'wb') as f_out:
            while True:
                chunk = f_in.read(PAGESIZE)
                if not chunk:
                    break
                f_out.write(chunk)
    shutil.rmtree(tempd)


def load_graph(path):
    with hdfs.open(path, 'rb') as f:
        serialized_graph_def = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(serialized_graph_def)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph


def save_graph(graph, path):
    with tf.Session(graph=graph):
        output_graph_def = graph.as_graph_def(add_shapes=True)
        serialized_graph_def = output_graph_def.SerializeToString()
    hdfs.dump(serialized_graph_def, path)


def add_jpeg_decoding(model, graph):
    """\
    Set up the JPEG decoding sub-graph.
    """
    m = model.copy()
    with tf.Session(graph=graph):
        jpeg_data = tf.placeholder(tf.string, name='jpeg_data')
        dimage = tf.image.decode_jpeg(jpeg_data, channels=m['input_depth'])
        dimage_as_float = tf.cast(dimage, dtype=tf.float32)
        dimage4d = tf.expand_dims(dimage_as_float, 0)
        resize_shape = tf.stack([m['input_height'], m['input_width']])
        resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
        resized_image = tf.image.resize_bilinear(dimage4d, resize_shape_as_int)
        offset_image = tf.subtract(resized_image, m['input_mean'])
        mul_image = tf.multiply(
            offset_image, 1.0 / m['input_std'], name='mul_image'
        )
    m['jpg_input_tensor_name'] = jpeg_data.name
    m['mul_image_tensor_name'] = mul_image.name
    return m


class BottleneckProjector(object):

    def __init__(self, model):
        self.graph = load_graph(model['prep_path'])
        self.jpg_input = self.graph.get_tensor_by_name(
            model['jpg_input_tensor_name']
        )
        self.mul_image = self.graph.get_tensor_by_name(
            model['mul_image_tensor_name']
        )
        self.bottleneck_tensor = self.graph.get_tensor_by_name(
            model['bottleneck_tensor_name']
        )
        self.resized_input_tensor = self.graph.get_tensor_by_name(
            model['resized_input_tensor_name']
        )

    def project(self, image_path):
        with hdfs.open(image_path, 'rb') as f:
            jpeg_data = f.read()
        with tf.Session(graph=self.graph) as s:
            resized_input = s.run(self.mul_image, {self.jpg_input: jpeg_data})
            bottleneck = s.run(self.bottleneck_tensor,
                               {self.resized_input_tensor: resized_input})
        return np.squeeze(bottleneck)


def add_training_and_evaluation(model, graph, n_classes, learning_rate,
                                export_dir):
    """\
    Add a softmax regression training layer and evaluation ops.

    Return: updated model info.
    """
    with tf.Session(graph=graph) as session:
        bneck_size = model['bottleneck_tensor_size']
        bneck_input = tf.placeholder(tf.float32, shape=[None, bneck_size])
        ground_truth_input = tf.placeholder(tf.float32, [None, n_classes])
        initial_value = tf.truncated_normal(
            [bneck_size, n_classes], stddev=0.001
        )
        layer_weights = tf.Variable(initial_value, name='final_weights')
        layer_biases = tf.Variable(tf.zeros([n_classes]), name='final_biases')
        logits = tf.matmul(bneck_input, layer_weights) + layer_biases
        final_tensor = tf.nn.softmax(logits, name="final_tensor")
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=ground_truth_input, logits=logits
        )
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean, name="train_step")
        prediction = tf.argmax(final_tensor, 1, name="prediction")
        correct_prediction = tf.equal(
            prediction, tf.argmax(ground_truth_input, 1)
        )
        eval_step = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name="evaluation_step"
        )
        session.run(tf.global_variables_initializer())
        tf.saved_model.simple_save(
            session, export_dir,
            {"bneck_input": bneck_input,
             "ground_truth_input": ground_truth_input},
            {"cross_entropy": cross_entropy_mean}
        )
    model = model.copy()
    model["train_step_op_name"] = train_step.name
    model["eval_step_tensor_name"] = eval_step.name
    return model


class Retrainer(object):

    # FIXME: we are getting metadata from a mix of built-in tf sources and our
    # own model info thing. We should be able to use the former for everything
    def __init__(self, model, export_dir):
        self.session = tf.InteractiveSession(graph=tf.Graph())
        metagraph = tf.saved_model.loader.load(
            self.session, [tag_constants.SERVING], export_dir
        )
        sig = metagraph.signature_def[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]
        graph = self.session.graph
        self.bottleneck_input = graph.get_tensor_by_name(
            sig.inputs["bneck_input"].name
        )
        self.ground_truth_input = graph.get_tensor_by_name(
            sig.inputs["ground_truth_input"].name
        )
        # should also be available as tf.get_collection("train_op")[0]
        self.train_step = graph.get_operation_by_name(
            model['train_step_op_name']
        )
        self.eval_step = graph.get_tensor_by_name(
            model['eval_step_tensor_name']
        )
        self.cross_entropy = graph.get_tensor_by_name(
            sig.outputs["cross_entropy"].name
        )

    def close_session(self):
        self.session.close()

    def run_train_step(self, bottlenecks, ground_truths):
        self.session.run(self.train_step, feed_dict={
            self.bottleneck_input: bottlenecks,
            self.ground_truth_input: ground_truths
        })

    def run_eval_step(self, bottlenecks, ground_truths):
        return self.session.run(
            [self.eval_step, self.cross_entropy],
            feed_dict={
                self.bottleneck_input: bottlenecks,
                self.ground_truth_input: ground_truths
            }
        )

    def run_validation_step(self, bottlenecks, ground_truths):
        return self.session.run(
            self.eval_step,
            feed_dict={
                self.bottleneck_input: bottlenecks,
                self.ground_truth_input: ground_truths
            }
        )

    def dump_output_graph(self, path):
        out_node_names = ["final_tensor"]  # FIXME
        out_graph_def = graph_util.convert_variables_to_constants(
            self.session, self.session.graph.as_graph_def(), out_node_names
        )
        hdfs.dump(out_graph_def.SerializeToString(), path)
