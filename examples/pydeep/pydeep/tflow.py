"""\
Create, store, load and execute graphs needed to retrain the network.

TODO: use the tf.data API
"""

import numpy as np
import pydoop.hdfs as hdfs
import tensorflow as tf
from tensorflow.python.framework import graph_util


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

    FINAL_TENSOR_NAME = "final_tensor"

    def __init__(self, model, n_classes, learning_rate):
        graph = model.load_prep()
        self.bneck_tensor = model.get_bottleneck(graph)
        self.jpg_input = model.get_jpg_input(graph)
        self.mul_image = model.get_mul_image(graph)
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
        self.final_tensor = tf.nn.softmax(logits, name=self.FINAL_TENSOR_NAME)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.ground_truth_input, logits=logits
        )
        self.cross_entropy_mean = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train_step = optimizer.minimize(
            self.cross_entropy_mean, name="train_step"
        )
        prediction = tf.argmax(self.final_tensor, 1, name="prediction")
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

    def dump_output_graph(self, path):
        out_node_names = [self.FINAL_TENSOR_NAME]
        out_graph_def = graph_util.convert_variables_to_constants(
            self.session, self.session.graph.as_graph_def(), out_node_names
        )
        hdfs.dump(out_graph_def.SerializeToString(), path)
