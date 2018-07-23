"""\
Minimal protocol for serializing many n-dimensional arrays of the same type.
"""

import numpy as np


class Writer(object):
    """\
    Writes arrays with the given shape and dtype. f must be open for writing
    bytes and metaf for writing text.
    """
    def __init__(self, f, metaf, shape, dtype):
        self.f = f
        meta = [np.dtype(dtype).str] + ["%d" % _ for _ in shape]
        metaf.write("\t".join(meta) + "\n")
        metaf.flush()
        self.shape = shape
        self.dtype = dtype

    def write(self, a):
        if a.shape != self.shape or a.dtype != self.dtype:
            raise ValueError("incompatible data type")
        self.f.write(a.tobytes())


class Reader(object):
    """\
    Reads arrays stored with arrayblob.Writer. f must be open for
    reading bytes and metaf for reading text.
    """
    def __init__(self, f, metaf):
        self.f = f
        meta = metaf.read().strip().split("\t")
        self.dtype = np.dtype(meta.pop(0))
        self.shape = tuple(int(_) for _ in meta)
        self.recsize = int(self.dtype.itemsize * np.prod(self.shape))
        self.count = 0
        self.__over = False

    def __next__(self):
        data = self.f.read(self.recsize)
        if not data:
            self.__over = True
        if self.__over:
            raise StopIteration
        self.count += 1
        return np.frombuffer(data, dtype=self.dtype).reshape(self.shape)

    def __iter__(self):
        return self

    def skip(self, n):
        if n < 0:
            raise ValueError("n must be positive")
        if n == 0:
            return
        self.count += n
        self.f.seek(self.count * self.recsize)
