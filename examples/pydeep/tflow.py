"""\
Create, store, load and execute graphs needed to retrain the network.

TODO: use the tf.data API
"""

import datetime
from mmap import PAGESIZE
import os
import shutil
import sys
import tarfile
import tempfile
import urllib

import numpy as np
import tensorflow as tf
import pydoop.hdfs as hdfs


# def get_model_graph(url, filename, dest_dir=None):
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


def load_graph(path, return_elements):
    with hdfs.open(path, 'rb') as f:
        serialized_graph_def = f.read()
    with tf.Graph().as_default() as graph:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(serialized_graph_def)
        elements = tf.import_graph_def(
            graph_def,
            name='',  # disable default 'import' prefix
            return_elements=return_elements
        )
    return graph, elements


def save_graph(graph, path):
    with tf.Session(graph=graph):
        output_graph_def = graph.as_graph_def(add_shapes=True)
        serialized_graph_def = output_graph_def.SerializeToString()
    with hdfs.open(path, 'wb') as f:
        f.write(serialized_graph_def)


class BottleneckProjector(object):

    @classmethod
    def create_graph(cls, model):
        """\
        Create a new (disjoint) graph that contains: a JPEG to
        normalized data conversion; and a graph constructed by cutting
        at the bottleneck an existing, pre-trained network.
        """
        def add_jpeg_decoding():
            m = model
            jpeg_data = tf.placeholder(tf.string, name=m['jpg_input'])
            dimage = tf.image.decode_jpeg(jpeg_data, channels=m['input_depth'])
            dimage_as_float = tf.cast(dimage, dtype=tf.float32)
            dimage_4d = tf.expand_dims(dimage_as_float, 0)
            resize_shape = tf.stack([m['input_height'], m['input_width']])
            resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
            resized_image = tf.image.resize_bilinear(dimage_4d,
                                                     resize_shape_as_int)
            offset_image = tf.subtract(resized_image, m['input_mean'])
            tf.multiply(offset_image, 1.0 / m['input_std'],
                        name=m['mul_image'])

        path = model['path']
        bneck_name = model['bottleneck_tensor_name']
        input_name = model['resized_input_tensor_name']
        graph, elements = load_graph(path, [bneck_name, input_name])
        with tf.Session(graph=graph):
            add_jpeg_decoding()
        return graph

    def __init__(self, model):
        self.model = model
        graph, (input_jpeg, mul_image, input_tensor, bneck_tensor) = \
            load_graph(model['path'], [model[x]
                                       for x in ['jpg_input_tensor_name',
                                                 'mul_image_tensor_name',
                                                 'resized_input_tensor_name',
                                                 'bottleneck_tensor_name']])
        self.graph = graph
        self.input_jpeg = input_jpeg
        self.mul_image = mul_image
        self.resized_input_tensor = input_tensor
        self.bneck_tensor = bneck_tensor

    def project(self, image_path):
        with hdfs.open(image_path, 'rb') as f:
            jpeg_data = f.read()
        with tf.Session(graph=self.graph) as s:
            m_idat = s.run(self.mul_image,
                           {self.input_jpeg: jpeg_data})
            b_val = s.run(self.bneck_tensor,
                          {self.resized_input_tensor: m_idat})
        return np.squeeze(b_val)


def variable_summaries(var):
    """\
    Attach summaries to Variable var (for TensorBoard).
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class Retrainer(object):

    @classmethod
    def create_graph(cls, model, n_classes):
        with tf.Graph().as_default() as graph:
            (ground_truth_input,
             final_tensor) = cls.add_training_ops(model, n_classes)
            cls.add_evaluation_step(final_tensor, ground_truth_input)
        return graph

    @classmethod
    def add_training_ops(cls, model, n_classes):
        """\
        Add a training graph.

        The graph will accept in input a bottleneck vector, with size
        defined in model, and a ground truth with n_classes possible
        options.
        """
        bneck_size = model['bottleneck_tensor_size']
        bneck_input = tf.placeholder(
            tf.float32, shape=[None, bneck_size], name='bottleneck_input')
        ground_truth_input = tf.placeholder(
            tf.float32, [None, n_classes], name='ground_truth_input')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # Organizing the following ops as `training_ops`
        # so they're easier to see in TensorBoard
        layer_name = 'training_ops'
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                initial_value = tf.truncated_normal(
                    [bneck_size, n_classes], stddev=0.001)
                layer_weights = tf.Variable(initial_value,
                                            name='final_weights')
                variable_summaries(layer_weights)
            with tf.name_scope('biases'):
                layer_biases = tf.Variable(tf.zeros([n_classes]),
                                           name='final_biases')
                variable_summaries(layer_biases)
            with tf.name_scope('Wx_plus_b'):
                logits = tf.matmul(bneck_input, layer_weights) + layer_biases
                tf.summary.histogram('pre_activations', logits)
        final_tensor = tf.nn.softmax(logits, name="final_tensor")
        tf.summary.histogram('activations', final_tensor)
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=ground_truth_input, logits=logits)
            with tf.name_scope('total'):
                cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer.minimize(cross_entropy_mean, name="train_step")
        return ground_truth_input, final_tensor

    @classmethod
    def add_evaluation_step(cls, result_tensor, ground_truth_tensor):
        """\
        Add operations needed to evaluate results accuracy.
        """
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                prediction = tf.argmax(result_tensor, 1, name="prediction")
                correct_prediction = tf.equal(
                    prediction, tf.argmax(ground_truth_tensor, 1))
            with tf.name_scope('accuracy'):
                evaluation_step = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32),
                    name="evaluation_step")
        tf.summary.scalar('accuracy', evaluation_step)

    def __init__(self, path):
        # we exploit the fact that we know what the graph should look like,
        # since it was built with our class method. Therefore, we can directly
        # extract the tensors we need.
        graph, (bottleneck_input, ground_truth_input,
                learning_rate, train_step, cross_entropy, final_tensor,
                evaluation_step, prediction) = \
            load_graph(path, ['bottleneck_input:0', 'ground_truth_input:0',
                              'learning_rate:0', 'train_step:0',
                              'cross_entropy:0', 'final_tensor:0',
                              'evaluation_step:0', 'prediction:0'])
        self.graph = graph
        self.bottleneck_input = bottleneck_input
        self.ground_truth_input = ground_truth_input
        self.learning_rate = learning_rate
        self.train_step = train_step
        self.cross_entropy = cross_entropy
        self.final_tensor = final_tensor
        self.evaluation_step = evaluation_step
        self.prediction = prediction

    def run_training_step(self, s, i, merged, bottlenecks, ground_truths):
        """\
        Feed the bottlenecks and ground truth into the graph, and run a
        training step. Capture training summaries for TensorBoard with
        the `merged` op.
        """
        summary, _ = s.run(
            [merged, self.train_step],
            feed_dict={self.bottleneck_input: bottlenecks,
                       self.ground_truth_input: ground_truths})
        return summary

    def run_check_training(self, s, i, bottlenecks, ground_truths):
        accuracy, cross_entropy_value = s.run(
            [self.evaluation_step, self.cross_entropy],
            feed_dict={self.bottleneck_input: bottlenecks,
                       self.ground_truth_input: ground_truths})
        tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                        (datetime.now(), i, accuracy * 100))
        tf.logging.info('%s: Step %d: Cross entropy = %f' %
                        (datetime.now(), i, cross_entropy_value))

    def run_validation(self, s, i, bottlenecks, ground_truths):
        summary, accuracy = s.run(
            [self.merged, self.evaluation_step],
            feed_dict={self.bottleneck_input: bottlenecks,
                       self.ground_truth_input: ground_truths})
        tf.logging.info(
            '%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
            (datetime.now(), i, accuracy * 100, len(bottlenecks)))
        return summary

    def run_test(self, s, bottlenecks, ground_truths, filenames,
                 do_print=False):
        accuracy, predictions = s.run(
            [self.evaluation_step, self.prediction],
            feed_dict={self.bottleneck_input: bottlenecks,
                       self.ground_truth_input: ground_truths})
        tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                        (accuracy * 100, len(bottlenecks)))
        if do_print:
            tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
            for i, filename in enumerate(filenames):
                if predictions[i] != ground_truths[i].argmax():
                    tf.logging.info('%70s  is %s predicts %s' %
                                    (filename, ground_truths[i].argmax(),
                                     predictions[i]))

    def run(self, FLAGS, get_random_bottlenecks):
        """\
        Run the training.

        get_random_bottlenecks is a function that takes the
        following input parameters:

         - category: "training", "testing" or "validation"
         - how_many: an integer that specifies how large the random
                     sample should be. If negative, return all bottlenecks.
        returns:

        - three lists with, respectively, bottleneck arrays, ground
          truths, and image filenames.
        """
        with tf.Session(graph=self.graph) as s:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(
                FLAGS.summaries_dir + '/train', s.graph)
            validation_writer = tf.summary.FileWriter(
                FLAGS.summaries_dir + '/validation')
            init = tf.global_variables_initializer()
            s.run(init)
            for i in range(FLAGS.how_many_training_steps):
                (train_bottlenecks,
                 train_ground_truths, _) = get_random_bottlenecks(
                     'training', FLAGS.train_batch_size)
                train_summary = self.run_training_step(
                    s, i, merged, train_bottlenecks, train_ground_truths)
                train_writer.add_summary(train_summary, i)
                # check from time to time
                is_last_step = (i + 1 == FLAGS.how_many_training_steps)
                if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                    self.run_check_training(s, i, train_bottlenecks,
                                            train_ground_truths)
                    (validation_bottlenecks, validation_ground_truth, _
                     ) = get_random_bottlenecks(
                         'validation', FLAGS.validation_batch_size)
                    validation_summary = self.run_validation(
                        s, i, validation_bottlenecks, validation_ground_truth)
                    validation_writer.add_summary(validation_summary, i)

            # final test
            (test_bottlenecks, test_ground_truth, test_filenames
             ) = get_random_bottlenecks('testing', FLAGS.test_batch_size)
            self.run_test(s, test_bottlenecks, test_ground_truth,
                          test_filenames,
                          FLAGS.print_misclassified_test_images)
