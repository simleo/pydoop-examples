"""\
Minimal protocol for serializing many n-dimensional arrays of the same type.
"""

import io

import numpy as np
import pydoop.hdfs as hdfsio


def write_meta(f, dtype, shape):
    meta = [np.dtype(dtype).str] + ["%d" % _ for _ in shape]
    f.write("\t".join(meta) + "\n")


def read_meta(f):
    meta = f.read().strip().split("\t")
    dtype = np.dtype(meta.pop(0))
    shape = tuple(int(_) for _ in meta)
    recsize = int(dtype.itemsize * np.prod(shape))
    return dtype, shape, recsize


class Writer(object):
    """\
    Writes arrays with the given shape and dtype. Creates a `name.data`
    binary data file and a `name.meta` text metadata file.
    """
    def __init__(self, name, shape, dtype, hdfs=True):
        open = hdfsio.open if hdfs else io.open
        with open("%s.meta" % name, "wt") as f:
            write_meta(f, dtype, shape)
        self.shape = shape
        self.dtype = dtype
        self.f = open("%s.data" % name, "wb")

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, a):
        if a.shape != self.shape or a.dtype != self.dtype:
            raise ValueError("incompatible data type")
        self.f.write(a.tobytes())


class Reader(object):
    """\
    Reads arrays stored with arrayblob.Writer. Looks for a `name.data`
    binary data file and a `name.meta` text metadata file.
    """
    def __init__(self, name, hdfs=True):
        open = hdfsio.open if hdfs else io.open
        with open("%s.meta" % name, "rt") as f:
            self.dtype, self.shape, self.recsize = read_meta(f)
        self.count = 0
        self.__over = False
        self.f = open("%s.data" % name, "rb")

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

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

    def to_npz(self, outf, labels=None):
        if labels:
            np.savez(outf, **dict(zip(labels, self)))
        else:
            np.savez(outf, *self)


def map_blobs(dir_path):
    m = {_["name"].rsplit("/", 1)[1]: _ for _ in hdfsio.lsl(dir_path)}
    rval = {}
    for basename, stat in m.items():
        if stat["kind"] != "file" or basename.startswith("_"):
            continue
        base, ext = hdfsio.path.splitext(basename)
        if ext == ".data":
            meta_bn = "%s.meta" % base
            if meta_bn not in m:
                raise RuntimeError("metafile for %r not found" % (basename,))
            with hdfsio.open(m[meta_bn]["name"], "rt") as f:
                dtype, shape, recsize = read_meta(f)
            n_records, r = divmod(stat["size"], recsize)
            assert r == 0
            rval[base] = n_records
    return rval
