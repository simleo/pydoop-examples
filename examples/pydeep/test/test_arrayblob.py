import unittest
import io

import numpy as np

import pydeep.arrayblob as arrayblob


class TestArrayBlob(unittest.TestCase):

    def setUp(self):
        n_arr = 8
        shape = (2, 3, 4)
        arrays = [np.random.randint(0, 100, shape) for _ in range(n_arr)]
        f, metaf = io.BytesIO(), io.StringIO()
        writer = arrayblob.Writer(f, metaf, arrays[0].shape, arrays[0].dtype)
        for a in arrays:
            writer.write(a)
        f.seek(0)
        metaf.seek(0)
        self.f, self.metaf, self.n_arr, self.arrays = f, metaf, n_arr, arrays

    def test_read(self):
        reader = arrayblob.Reader(self.f, self.metaf)
        recs = [_ for _ in reader]
        self.assertEqual(len(recs), self.n_arr)
        for r, a in zip(recs, self.arrays):
            self.assertTrue(np.array_equal(r, a))

    def test_skip(self):
        reader = arrayblob.Reader(self.f, self.metaf)
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
        writer = arrayblob.Writer(io.BytesIO(), io.StringIO(), shape, dtype)
        writer.write(a)
        with self.assertRaises(ValueError):
            writer.write(a.reshape((shape[1], shape[0])))
        with self.assertRaises(ValueError):
            writer.write(a.astype("f2"))
        reader = arrayblob.Reader(self.f, self.metaf)
        with self.assertRaises(ValueError):
            reader.skip(-1)


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestArrayBlob)


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
