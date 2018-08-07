import os
import shutil
import tempfile
import unittest

import numpy as np

import pydeep.arrayblob as arrayblob


class TestArrayBlob(unittest.TestCase):

    def setUp(self):
        self.n_arr = 8
        shape = (2, 3, 4)
        self.arrays = [np.random.randint(0, 100, shape)
                       for _ in range(self.n_arr)]
        dtype = self.arrays[0].dtype
        self.wd = tempfile.mkdtemp(prefix="pydeep_")
        self.name = os.path.join(self.wd, "blob")
        with arrayblob.Writer(self.name, shape, dtype, hdfs=False) as writer:
            for a in self.arrays:
                writer.write(a)

    def tearDown(self):
        shutil.rmtree(self.wd)

    def test_read(self):
        with arrayblob.Reader(self.name, hdfs=False) as reader:
            recs = [_ for _ in reader]
        self.assertEqual(len(recs), self.n_arr)
        for r, a in zip(recs, self.arrays):
            self.assertTrue(np.array_equal(r, a))

    def test_skip(self):
        with arrayblob.Reader(self.name, hdfs=False) as reader:
            self.assertTrue(np.array_equal(next(reader), self.arrays[0]))
            reader.skip(3)
            self.assertTrue(np.array_equal(next(reader), self.arrays[4]))
            reader.skip(1)
            self.assertTrue(np.array_equal(next(reader), self.arrays[6]))
            reader.skip(self.n_arr)
            with self.assertRaises(StopIteration):
                next(reader)

    def test_errors(self):
        shape, dtype = (2, 3), "i1"
        a = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        with arrayblob.Writer(self.name + "_", shape, dtype) as writer:
            writer.write(a)
            with self.assertRaises(ValueError):
                writer.write(a.reshape((shape[1], shape[0])))
            with self.assertRaises(ValueError):
                writer.write(a.astype("f2"))
        with arrayblob.Reader(self.name, hdfs=False) as reader:
            with self.assertRaises(ValueError):
                reader.skip(-1)

    def test_npz(self):
        with arrayblob.Reader(self.name, hdfs=False) as reader:
            out_fn = "%s.npz" % self.name
            with open(out_fn, "wb") as outf:
                reader.to_npz(outf)
        self.__check_npz(out_fn)

    def test_npz_labels(self):
        with arrayblob.Reader(self.name, hdfs=False) as reader:
            labels = ["foo_%d" % _ for _ in range(self.n_arr)]
            out_fn = "%s.npz" % self.name
            with open(out_fn, "wb") as outf:
                reader.to_npz(outf, labels=labels)
        self.__check_npz(out_fn, labels=labels)

    def __check_npz(self, out_fn, labels=None):
        npz = np.load(out_fn)
        keys = npz.keys()
        if labels:
            self.assertEqual(list(keys), labels)
        else:
            self.assertEqual(len(keys), self.n_arr)
        for i, k in enumerate(keys):
            self.assertTrue(np.array_equal(npz[k], self.arrays[i]))


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestArrayBlob)


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
