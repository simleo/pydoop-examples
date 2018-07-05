import unittest

import pydeep.common as common


class TestCommon(unittest.TestCase):

    def test_balanced_split(self):
        for seq_len in 15, 16, 17, 18, 19, 20:
            seq, N = list(range(seq_len)), 4
            groups = list(common.balanced_split(seq, N))
            self.assertEqual(len(groups), N)
            self.assertEqual(sum(groups, []), seq)
            sg = sorted(groups, key=len)
            self.assertTrue(len(sg[-1]) - len(sg[0]) <= 1)

    def test_balanced_split_errors(self):
        for N in -1, 0:
            with self.assertRaises(ValueError):
                list(common.balanced_split([42], N))
        with self.assertRaises(ValueError):
            list(common.balanced_split([42], 2))
        with self.assertRaises(ValueError):
            list(common.balanced_split([], 1))


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestCommon)


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
