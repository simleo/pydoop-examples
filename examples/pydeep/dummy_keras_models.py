"""\
Generate dummy Keras models for functional testing.
"""

import argparse
import os
import shutil
import sys

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

import pydeep.arrayblob as arrayblob

BATCH_SIZE = 32
EPOCHS = 1
LOSS = "categorical_crossentropy"
METRICS = ["accuracy"]
OPTIMIZER = RMSprop()
NUM_CLASSES = 10  # expected


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", metavar="OUTPUT_DIR")
    parser.add_argument("--n-models", metavar='INT', type=int, default=4)
    return parser


def shrink(a, factor):
    return a[np.random.randint(0, a.shape[0], a.shape[0] // factor)]


def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert len(set(y_train.flat)) == len(set(y_test.flat)) == NUM_CLASSES
    data = {
        "x": {"train": x_train, "test": x_test},
        "y": {"train": y_train, "test": y_test},
    }
    for s in "x", "y":
        for cat in "train", "test":
            data[s][cat] = shrink(data[s][cat], 10)
    for cat in "train", "test":
        assert len(set(data["y"][cat].flat)) == NUM_CLASSES
    for k in "train", "test":
        new_shape = (data["x"][k].shape[0], np.prod(data["x"][k].shape[1:]))
        M = np.iinfo(data["x"][k].dtype).max
        data["x"][k] = data["x"][k].reshape(new_shape).astype(np.float32) / M
        data["y"][k] = keras.utils.to_categorical(data["y"][k], NUM_CLASSES)
    return data


def build_model(input_shape):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    return model


def train_and_dump_models(data, n_models, output_dir):
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    for i in range(n_models):
        base = os.path.join(models_dir, "model_%d" % i)
        print("training %r..." % (base,))
        model = build_model((data["x"]["train"].shape[1],))
        model.fit(
            data["x"]["train"], data["y"]["train"],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=False,
            validation_data=(data["x"]["test"], data["y"]["test"])
        )
        with open("%s.json" % base, "wt") as f:
            f.write(model.to_json())
        model.save_weights("%s.hdf5" % base)
    shutil.make_archive(
        os.path.join(output_dir, "models"), "tar", root_dir=models_dir
    )
    shutil.rmtree(models_dir)


def save_test_data(test_data, output_dir):
    blob_base = os.path.join(output_dir, "mnist_test")
    print("saving test data as %r" % (blob_base,))
    shape, dtype = test_data[0].shape, test_data[0].dtype
    with arrayblob.Writer(blob_base, shape, dtype, hdfs=False) as writer:
        for a in test_data:
            writer.write(a)


def main(argv=sys.argv):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = build_parser().parse_args(argv[1:])
    data = get_data()
    for s in "x", "y":
        for cat in "train", "test":
            print("%s_%s: %r" % (s, cat, data[s][cat].shape))
    train_and_dump_models(data, args.n_models, args.output)
    save_test_data(data["x"]["test"], args.output)


if __name__ == "__main__":
    sys.exit(main())
