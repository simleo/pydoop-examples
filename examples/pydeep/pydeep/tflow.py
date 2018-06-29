"""\
Create, store, load and execute graphs needed to retrain the network.

TODO: use the tf.data API
"""

import numpy as np
import pydoop.hdfs as hdfs
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2

from tensorflow.python.framework import tensor_util


class BottleneckProjector(object):

    def __init__(self, model):
        graph = model.load_prep()
        self.jpg_input = model.get_jpg_input(graph)
        self.mul_image = model.get_mul_image(graph)
        self.bottleneck_tensor = model.get_bottleneck(graph)
        self.resized_input_tensor = model.get_resized_input(graph)
        self.session = tf.InteractiveSession(graph=graph)

    def close_session(self):
        self.session.close()

    def project(self, jpeg_data):
        resized_input = self.session.run(
            self.mul_image, {self.jpg_input: jpeg_data}
        )
        bottleneck = self.session.run(
            self.bottleneck_tensor, {self.resized_input_tensor: resized_input}
        )
        return np.squeeze(bottleneck)


class Retrainer(object):

    def __init__(self, model, n_classes, learning_rate):
        graph = model.load_prep()
        self.bneck_tensor = model.get_bottleneck(graph)
        self.session = tf.InteractiveSession(graph=graph)
        self.__add_train_and_eval(n_classes, learning_rate)

    def close_session(self):
        self.session.close()

    def __add_train_and_eval(self, n_classes, learning_rate):
        """\
        Add a softmax regression training layer and evaluation ops.

        Note that the tf.Variable and optimizer.minimize calls add standard
        collections to the graph under the hood ('train_op', 'variables' and
        'trainable_variables').
        """
        bneck_size = self.bneck_tensor.shape[1].value
        self.bneck_input = tf.placeholder(
            self.bneck_tensor.dtype, shape=[None, bneck_size]
        )
        self.ground_truth_input = tf.placeholder(tf.float32, [None, n_classes])
        initial_value = tf.truncated_normal(
            [bneck_size, n_classes], stddev=0.001
        )
        layer_weights = tf.Variable(initial_value, name='final_weights')
        layer_biases = tf.Variable(tf.zeros([n_classes]), name='final_biases')
        logits = tf.matmul(self.bneck_input, layer_weights) + layer_biases
        final_tensor = tf.nn.softmax(logits, name="final_tensor")
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.ground_truth_input, logits=logits
        )
        self.cross_entropy_mean = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train_step = optimizer.minimize(
            self.cross_entropy_mean, name="train_step"
        )
        prediction = tf.argmax(final_tensor, 1, name="prediction")
        correct_prediction = tf.equal(
            prediction, tf.argmax(self.ground_truth_input, 1)
        )
        self.eval_step = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name="evaluation_step"
        )
        self.session.run(tf.global_variables_initializer())

    def run_train_step(self, bottlenecks, ground_truths):
        self.session.run(self.train_step, feed_dict={
            self.bneck_input: bottlenecks,
            self.ground_truth_input: ground_truths
        })

    def run_eval_step(self, bottlenecks, ground_truths):
        return self.session.run(
            [self.eval_step, self.cross_entropy_mean],
            feed_dict={
                self.bneck_input: bottlenecks,
                self.ground_truth_input: ground_truths
            }
        )

    def run_validation_step(self, bottlenecks, ground_truths):
        return self.session.run(
            self.eval_step,
            feed_dict={
                self.bneck_input: bottlenecks,
                self.ground_truth_input: ground_truths
            }
        )

    def __freeze_vars(self):
        """\
        Adapted from tf.graph_util.convert_variables_to_constants.
        """
        graph = self.session.graph
        var_names = [
            _.name for _ in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        ]
        var_values = self.session.run(var_names)
        var_map = dict(zip(var_names, var_values))
        output_graph_def = graph_pb2.GraphDef()
        input_graph_def = graph.as_graph_def()
        for input_node in input_graph_def.node:
            output_node = node_def_pb2.NodeDef()
            if input_node.name in var_map:
                output_node.op = "Const"
                output_node.name = input_node.name
                dtype = input_node.attr["dtype"]
                data = var_map[input_node.name]
                output_node.attr["dtype"].CopyFrom(dtype)
                output_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(
                            data, dtype=dtype.type, shape=data.shape)))
            else:
                output_node.CopyFrom(input_node)
            output_graph_def.node.extend([output_node])
        output_graph_def.library.CopyFrom(input_graph_def.library)
        return output_graph_def

    def dump_output_graph(self, path):
        out_graph_def = self.__freeze_vars()
        meta_graph_def = tf.train.export_meta_graph(graph_def=out_graph_def)
        hdfs.dump(meta_graph_def.SerializeToString(), path)
