from hashlib import md5
import unittest
import uuid

import numpy as np
import pydoop.hdfs as hdfs

from pydeep.ioformats import BottleneckStore


CHECKSUM_SIZE = md5().digest_size


def random_checksum():
    return np.random.randint(0, 256, CHECKSUM_SIZE, dtype=np.uint8).tobytes()


class TestBottleneckStore(unittest.TestCase):

    def setUp(self):
        self.wd = "pydeep_%s" % uuid.uuid4().hex
        classes = "a", "b", "c"
        files_per_class = 2
        bnecks_per_file = 3
        self.bneck_len = 8
        self.bneck_dtype = np.int32
        hdfs.mkdir(self.wd)
        self.bneck_map = {}
        for c in classes:
            self.bneck_map[c] = []
            subd = hdfs.path.join(self.wd, c)
            hdfs.mkdir(subd)
            for i in range(files_per_class):
                with hdfs.open(hdfs.path.join(subd, "f%s" % i), "wb") as f:
                    for j in range(bnecks_per_file):
                        checksum = random_checksum()
                        bneck = self.__random_bneck()
                        f.write(checksum + bneck.tobytes())
                        self.bneck_map[c].append((checksum, bneck))
        self.store = BottleneckStore(self.wd, self.bneck_len, self.bneck_dtype)

    def tearDown(self):
        hdfs.rmr(self.wd)

    def test_get_all_bnecks(self):
        self.__check_bnecks(self.store.get_all_bnecks())

    def test_get_bnecks(self):
        self.__check_bnecks(self.store.get_bnecks(checksums=True))

    def __random_bneck(self):
        return np.random.randint(0, 99, self.bneck_len, dtype=self.bneck_dtype)

    def __check_bnecks(self, bneck_map):
        self.assertEqual(set(bneck_map), set(self.bneck_map))
        for c, bnecks in bneck_map.items():
            bnecks = sorted(bnecks)
            exp_bnecks = sorted(self.bneck_map[c])
            for (cs, bn), (exp_cs, exp_bn) in zip(bnecks, exp_bnecks):
                self.assertEqual(cs, exp_cs)
                self.assertTrue(np.array_equal(bn, exp_bn))


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestBottleneckStore)


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
