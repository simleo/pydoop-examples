import unittest

import pydeep.common as common


class TestCommon(unittest.TestCase):

    def setUp(self):
        self.N = 4
        self.lengths = [15, 16, 17, 18, 19, 20]

    def test_balanced_parts(self):
        for L in self.lengths:
            parts = common.balanced_parts(L, self.N)
            self.assertEqual(len(parts), self.N)
            self.assertEqual(sum(parts), L)
            parts.sort()
            self.assertLessEqual(parts[-1] - parts[0], 1)

    def test_balanced_chunks(self):
        for L in self.lengths:
            chunks = list(common.balanced_chunks(L, self.N))
            self.assertEqual(len(chunks), self.N)
            for i in range(self.N - 1):
                self.assertEqual(sum(chunks[i]), chunks[i + 1][0])
            parts = sorted(_[1] for _ in chunks)
            self.assertEqual(sum(parts), L)
            self.assertLessEqual(parts[-1] - parts[0], 1)

    def test_balanced_split(self):
        for L in self.lengths:
            seq = list(range(L))
            groups = list(common.balanced_split(seq, self.N))
            self.assertEqual(len(groups), self.N)
            self.assertEqual(sum(groups, []), seq)
            sg = sorted(groups, key=len)
            self.assertTrue(len(sg[-1]) - len(sg[0]) <= 1)

    def test_errors(self):
        # N < 1
        for N in -1, 0:
            with self.assertRaises(ValueError):
                common.balanced_parts(1, N)
            with self.assertRaises(ValueError):
                list(common.balanced_chunks(1, N))
            with self.assertRaises(ValueError):
                list(common.balanced_split([42], N))
        # N > L
        with self.assertRaises(ValueError):
            common.balanced_parts(1, 2)
        with self.assertRaises(ValueError):
            list(common.balanced_chunks(1, 2))
        with self.assertRaises(ValueError):
            list(common.balanced_split([42], 2))


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(TestCommon)


if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(suite())
